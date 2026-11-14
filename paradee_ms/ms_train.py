"""Compact trainers for the Paradee text side and decoder on row shards (continue.plan.v4.md C2, C3).

  ms_train.py dec  --init init/dec_row0.pt  --data data/teacher --held 4 --steps 1500 --max-minutes 8 --out ckpt/dec
  ms_train.py text --init init/text_row0.pt --data data/teacher --held 4 --steps  800 --max-minutes 4 --out ckpt/text

Losses are Paradee's (train_text.py / train_vhead.py at commit 2de547e), spectral stage only.
Rows: the first --held rows (in shard order) are held out, never trained on.
"""
import argparse
import glob
import json
import math
import os
import random
import sys
import time
import warnings

warnings.filterwarnings('ignore')
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torchaudio  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from student import TextStudent, alignment  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('which', choices=['dec', 'text'])
ap.add_argument('--init', required=True)
ap.add_argument('--data', default='data/teacher')
ap.add_argument('--held', type=int, default=4)
ap.add_argument('--steps', type=int, default=1000)
ap.add_argument('--bs', type=int, default=16)
ap.add_argument('--lr', type=float, default=None)
ap.add_argument('--seg', type=int, default=64)
ap.add_argument('--max-minutes', type=float, default=1e9)
ap.add_argument('--out', required=True)
ap.add_argument('--smooth', action='store_true', help='median/moving-average the F0 and N targets (text side)')
ap.add_argument('--aug', type=float, default=0.0, help='decoder: strength of smooth F0/N jitter on the inputs (0 = off)')
ap.add_argument('--eval-every', type=int, default=250)
ap.add_argument('--patience', type=int, default=8)
ap.add_argument('--wd', type=float, default=0.01)
ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
A = ap.parse_args()
DEV = A.device
os.makedirs(A.out, exist_ok=True)

rows = []
for p in sorted(glob.glob(os.path.join(A.data, 'shard_*.pt'))):
    rows += torch.load(p)
if A.smooth:
    import numpy as _np
    from scipy.signal import medfilt
    for _r in rows:
        _r['F0'] = torch.from_numpy(medfilt(_r['F0'].numpy(), 5).astype('float32'))
        _r['N'] = torch.from_numpy(_np.convolve(_np.pad(_r['N'].numpy(), 2, mode='edge'), _np.ones(5) / 5, mode='valid').astype('float32'))
held, train = rows[:A.held], rows[A.held:]
print(f'{len(train)} train / {len(held)} held-out rows', flush=True)
T0 = time.time()
log = open(os.path.join(A.out, 'log.jsonl'), 'a')


def cosine(step, total, warm=50):
    return min(1, (step + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(step, total) / total))


# ================================================================ text side
if A.which == 'text':
    LR = A.lr or 5e-4

    def collate(batch):
        B = len(batch)
        T = max(len(r['ids']) for r in batch)
        Fm = max(len(r['F0']) for r in batch) // 2
        ids = torch.zeros(B, T, dtype=torch.long)
        L = torch.tensor([len(r['ids']) for r in batch])
        dur = torch.zeros(B, T)
        ten = torch.zeros(B, 512, T)
        aln = torch.zeros(B, T, Fm)
        F0 = torch.zeros(B, 2 * Fm)
        N = torch.zeros(B, 2 * Fm)
        fm = torch.zeros(B, 2 * Fm, dtype=torch.bool)
        for i, r in enumerate(batch):
            t = len(r['ids'])
            f = len(r['F0']) // 2
            ids[i, :t] = r['ids']
            dur[i, :t] = r['dur']
            ten[i, :, :t] = r['t_en'].float()
            aln[i, :t, :f] = alignment(r['pred_dur'], f)
            F0[i, :2 * f] = r['F0']
            N[i, :2 * f] = r['N']
            fm[i, :2 * f] = True
        tm = torch.arange(T)[None] < L[:, None]
        return [x.to(DEV) for x in (ids, L, dur, ten, aln, F0, N, tm, fm)]

    def losses(model, batch):
        ids, L, dur, ten, aln, F0, N, tm, fm = batch
        pd_, pF0, pN, pasr = model(ids, L, aln)
        l_dur = (F.l1_loss(torch.log(pd_.clamp_min(1e-3)), torch.log(dur.clamp_min(1e-3)), reduction='none') * tm).sum() / tm.sum()
        l_f0 = (F.l1_loss(pF0, F0, reduction='none') * fm).sum() / fm.sum() / 100
        l_n = (F.l1_loss(pN, N, reduction='none') * fm).sum() / fm.sum()
        l_asr = (F.l1_loss(pasr, ten, reduction='none') * tm[:, None]).sum() / tm.sum() / 512
        return l_dur, l_f0, l_n, l_asr

    model = TextStudent('s+mlp').to(DEV)
    model.load_state_dict(torch.load(A.init, map_location=DEV)['model'])
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=A.wd)

    @torch.no_grad()
    def evaluate():
        model.eval()
        ls = [l.item() for l in losses(model, collate(held))]
        # free-running: predicted durations against the measured ones
        mae, ratio = [], []
        for r in held:
            ids = r['ids'][None].to(DEV)
            L = torch.tensor([len(r['ids'])], device=DEV)
            pd_, _, _, _ = model(ids, L)
            p = torch.round(pd_[0]).clamp(min=1).cpu()
            mae.append((p - r['pred_dur'].float()).abs().mean().item())
            ratio.append(float(p.sum() / r['pred_dur'].sum()))
        model.train()
        return dict(l_dur=ls[0], l_f0=ls[1], l_n=ls[2], l_asr=ls[3], dur_mae_frames=sum(mae) / len(mae),
                    length_ratio=sum(ratio) / len(ratio))

    step = 0
    best = [float('inf'), 0, None]
    model.train()
    print('before:', json.dumps(evaluate()), flush=True)
    while step < A.steps and (time.time() - T0) / 60 < A.max_minutes:
        batch = collate(random.sample(train, min(A.bs, len(train))))
        for g in opt.param_groups:
            g['lr'] = LR * cosine(step, A.steps)
        ls = losses(model, batch)
        loss = sum(ls)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        if step % A.eval_every == 0:
            ev_ = evaluate()
            sc_ = ev_['l_dur'] + ev_['l_f0'] + ev_['l_n'] + ev_['l_asr']
            print('EVAL', step, json.dumps({k: round(v, 3) for k, v in ev_.items()}), 'score', round(sc_, 3), flush=True)
            if sc_ < best[0]:
                best[:] = [sc_, step, {k: v.detach().clone() for k, v in model.state_dict().items()}]
            elif step - best[1] >= A.patience * A.eval_every:
                print('early stop, best step', best[1], flush=True)
                break
        if step % 100 == 0:
            print(f'step {step} loss {loss.item():.4f} dur {ls[0].item():.3f} f0 {ls[1].item():.3f} n {ls[2].item():.3f} asr {ls[3].item():.3f} {(time.time() - T0) / step:.2f}s/step', flush=True)
    if best[2] is not None:
        model.load_state_dict(best[2])
    ev = evaluate()
    ev.update(step=best[1] or step, trained_steps=step, minutes=(time.time() - T0) / 60)
    print('after:', json.dumps(ev), flush=True)
    log.write(json.dumps(ev) + '\n')
    torch.save({'model': model.state_dict(), 'step': step}, os.path.join(A.out, 'last.pt'))
    sys.exit(0)

# ================================================================ decoder
from vhead import VDecoder  # noqa: E402

LR = A.lr or 2e-4


def full(r):
    aln = alignment(r['pred_dur'])
    asr = r['t_en'].float() @ aln
    Fr = aln.shape[1]
    return asr, r['F0'][:2 * Fr], r['N'][:2 * Fr], r['audio'][:600 * Fr].float() / 32767


def segment(r, seg):
    asr, F0, N, au = full(r)
    Fr = asr.shape[1]
    if Fr <= seg:
        pad = seg - Fr
        asr = F.pad(asr, (0, pad))
        F0 = F.pad(F0, (0, 2 * pad))
        N = F.pad(N, (0, 2 * pad))
        au = F.pad(au, (0, 600 * pad))
        a = 0
    else:
        a = random.randint(0, Fr - seg)
    c0 = (F0[:2 * a].double() * 300 / 24000).sum() % 1
    return asr[:, a:a + seg], F0[2 * a:2 * (a + seg)], N[2 * a:2 * (a + seg)], au[600 * a:600 * (a + seg)], c0.float()


def collate_dec(batch, seg):
    x = [segment(r, seg) for r in batch]
    return [torch.stack([t[i] for t in x]).to(DEV) for i in range(5)]


mel = torchaudio.transforms.MelSpectrogram(24000, n_fft=1024, hop_length=256, n_mels=80).to(DEV)


def spec_losses(y, t):
    lm = F.l1_loss(torch.log(mel(y).clamp_min(1e-5)), torch.log(mel(t).clamp_min(1e-5)))
    ls = 0
    SZ = (64, 128, 512, 1024, 2048)
    for n in SZ:
        w = torch.hann_window(n, device=DEV)
        Y = torch.stft(y, n, n // 4, window=w, return_complex=True).abs()
        T_ = torch.stft(t, n, n // 4, window=w, return_complex=True).abs()
        ls = ls + (T_ - Y).norm() / T_.norm().clamp_min(1e-5) + F.l1_loss(torch.log(Y.clamp_min(1e-5)), torch.log(T_.clamp_min(1e-5)))
    return lm, ls / len(SZ)


def harm_loss(y, t, f0, n=2048, hop=512):
    w = torch.hann_window(n, device=DEV)
    nfr = y.shape[-1] // hop + 1
    idx = (torch.arange(nfr, device=DEV) * hop // 300).clamp(max=f0.shape[1] - 1)
    lo = (idx - 2).clamp(min=0)
    hi = (idx + 2).clamp(max=f0.shape[1] - 1)
    f = f0[:, idx]
    v = (torch.minimum(torch.minimum(f0[:, lo], f), f0[:, hi]) > 100).float()
    fr = torch.arange(n // 2 + 1, device=DEV) * 24000 / n
    ratio = fr[None, :, None] / f.clamp_min(60)[:, None]
    frac = (ratio - ratio.round()).abs()
    mh, mi = (frac < 0.15).float(), ((frac - 0.5).abs() < 0.15).float()
    bands = torch.stack([((fr >= 1000 * b) & (fr < 1000 * (b + 1))).float() for b in range(1, 9)])
    out = []
    for x in (y, t):
        P = torch.stft(x, n, hop, window=w, return_complex=True).abs().pow(2)
        out.append(torch.log((torch.einsum('bft,kf->bkt', P * mh, bands) + 1e-8) / (torch.einsum('bft,kf->bkt', P * mi, bands) + 1e-8)))
    return ((out[0] - out[1]).abs() * v[:, None]).sum() / (v.sum() * 8).clamp_min(1)


MODB = [(4, 8), (8, 12), (12, 16), (16, 21), (21, 29)]


def mod_loss(y, t, cyc, uv, n=64, hop=16, win=75):
    w = torch.hann_window(n, device=DEV)
    k = y.shape[-1] // hop + 1
    ph = 2 * torch.pi * F.pad(cyc, (0, hop))[:, ::hop][:, :k]
    v = F.pad(uv, (0, hop))[:, ::hop][:, :k]
    carr = torch.stack([fn(mm * ph) for mm in (1, 2) for fn in (torch.cos, torch.sin)], 1)
    D = []
    for x in (y, t):
        X = torch.stft(x, n, hop, window=w, return_complex=True).abs()
        e = torch.stack([X[:, a:b].sum(1) for a, b in MODB], 1)
        num = F.avg_pool1d((e[:, :, None] * carr[:, None]).flatten(1, 2), win, win // 2)
        den = F.avg_pool1d(e, win, win // 2).clamp_min(1e-5)
        D.append(2 * num.view(e.shape[0], 5, 4, -1) / den[:, :, None])
    vm = (F.avg_pool1d(v[:, None], win, win // 2) > 0.99).float()[:, :, None]
    return ((D[0] - D[1]).abs() * vm).sum() / (vm.sum() * 20).clamp_min(1)


model = VDecoder(head='v2', dim=160, blocks=4).to(DEV)
model.load_ckpt(A.init, DEV)
opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.8, 0.99), weight_decay=A.wd)


def jitter(x, sigma, voiced=None):
    """Smooth random multiplicative (voiced) / additive jitter, a few hundred ms wide."""
    B, L = x.shape
    z = F.interpolate(torch.randn(B, 1, max(2, L // 24), device=x.device), size=L, mode='linear', align_corners=False)[:, 0]
    return z * sigma


@torch.no_grad()
def evaluate(save=False):
    model.eval()
    tot, n = 0.0, 0
    for i, r in enumerate(held):
        asr, F0, N, au = full(r)
        torch.manual_seed(i)
        y = model(asr[None].to(DEV), F0[None].to(DEV), N[None].to(DEV)).squeeze().cpu()
        k = min(len(y), len(au))
        lm, ls = spec_losses(y[None, :k].to(DEV), au[None, :k].to(DEV))
        tot += float(lm + ls)
        n += 1
        if save:
            sf.write(os.path.join(A.out, f'held_{i + 1}_student.wav'), y.numpy(), 24000)
            sf.write(os.path.join(A.out, f'held_{i + 1}_target.wav'), au.numpy(), 24000)
    model.train()
    return dict(held_spec_loss=tot / max(n, 1))


step = 0
model.train()
before = evaluate()
print('before:', json.dumps(before), flush=True)
while step < A.steps and (time.time() - T0) / 60 < A.max_minutes:
    asr, F0, N, au, c0 = collate_dec(random.sample(train, min(A.bs, len(train))) if len(train) >= A.bs else [random.choice(train) for _ in range(A.bs)], A.seg)
    for g in opt.param_groups:
        g['lr'] = LR * cosine(step, A.steps)
    if A.aug > 0:
        F0_in = F0 * torch.exp(jitter(F0, 0.08 * A.aug)) * (F0 > 0)
        N_in = N + jitter(N, 0.4 * A.aug)
    else:
        F0_in, N_in = F0, N
    y = model(asr, F0_in, N_in, c0=c0)
    lm, ls = spec_losses(y, au)
    lh = harm_loss(y, au, F0)
    lmod = mod_loss(y, au, model.head.last['cyc'], model.head.last['uv'])
    loss = lm + ls + 2.0 * lh + 5.0 * lmod
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    opt.step()
    step += 1
    if step % 100 == 0:
        print(f'step {step} loss {loss.item():.4f} mel {lm.item():.3f} stft {ls.item():.3f} harm {lh.item():.3f} mod {lmod.item():.3f} {(time.time() - T0) / step:.2f}s/step', flush=True)
    if step % 500 == 0:
        print('EVAL', json.dumps(evaluate()), flush=True)
after = evaluate(save=True)
after.update(step=step, minutes=(time.time() - T0) / 60, before=before['held_spec_loss'])
print('after:', json.dumps(after), flush=True)
log.write(json.dumps(after) + '\n')
torch.save({'model': model.state_dict(), 'step': step}, os.path.join(A.out, 'last.pt'))
