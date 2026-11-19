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
ap.add_argument('--score', choices=['sum', 'dur_asr'], default='sum', help='text side: which held-out losses pick the checkpoint')
ap.add_argument('--uv-text', action='store_true', help='text side: log-F0 + voiced/unvoiced head (REVIEW Phase 1)')
ap.add_argument('--uv-unvoiced-weight', type=float, default=0.1, help='UV text: l_f0 weight on unvoiced frames (Gate 1 retry uses 0.0)')
ap.add_argument('--uv-uv-weight', type=float, default=1.0, help='UV text: l_uv weight (Gate 1 retry uses 2.0)')
ap.add_argument('--gan', action='store_true', help='decoder: adversarial stage (Paradee stage two)')
ap.add_argument('--gan-warmup', type=int, default=0, help='GAN: update D only for this many steps first (REVIEW Phase 3: 500)')
ap.add_argument('--dlr', type=float, default=2e-4, help='GAN: discriminator learning rate')
ap.add_argument('--gta-p', type=float, default=0.0, help='decoder: probability of using the text model\'s own (GTA) F0/N/asr inputs (REVIEW Phase 2)')
ap.add_argument('--mel-weight', type=float, default=3.0)
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
    UV = A.uv_text

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

    if UV:
        from student_uv import TextStudentUV, set_lf0_stats, reinit_f0_proj
        model = TextStudentUV('s+mlp').to(DEV)
        model.load_state_dict(torch.load(A.init, map_location=DEV, weights_only=False)['model'], strict=False)
        reinit_f0_proj(model)                 # old F0_proj emitted Hz, new one predicts normalised log-F0
        set_lf0_stats(model, train)           # mu/sd of log-F0 over voiced training frames
        print(f'UV text side: lf0 mu/sd {model.lf0_mu.item():.3f}/{model.lf0_sd.item():.3f}', flush=True)

        def lf0_target(F0hz):
            """v = F0>0; log-F0 linearly interpolated across unvoiced gaps (edges held constant) -> normalised."""
            import numpy as np
            f = F0hz.detach().cpu().numpy()
            B, Fr = f.shape
            v = f > 0
            lf0c = np.empty_like(f)
            idx = np.arange(Fr)
            mu = float(model.lf0_mu)
            for b in range(B):
                vi = np.where(v[b])[0]
                if len(vi) == 0:
                    lf0c[b] = mu
                else:
                    lf0c[b] = np.interp(idx, vi, np.log(f[b, vi]))  # holds edges constant
            lf0c = torch.as_tensor(lf0c, device=F0hz.device, dtype=F0hz.dtype)
            return (lf0c - model.lf0_mu) / model.lf0_sd, torch.as_tensor(v, device=F0hz.device, dtype=F0hz.dtype)

        def losses(model, batch):
            ids, L, dur, ten, aln, F0, N, tm, fm = batch
            pd_, plf0, puv, pN, pasr = model(ids, L, aln)
            tgt_lf0, tgt_uv = lf0_target(F0)
            w = torch.where(tgt_uv > 0.5, 1.0, A.uv_unvoiced_weight)
            l_dur = (F.l1_loss(torch.log(pd_.clamp_min(1e-3)), torch.log(dur.clamp_min(1e-3)), reduction='none') * tm).sum() / tm.sum()
            l_f0 = (F.l1_loss(plf0, tgt_lf0, reduction='none') * fm * w).sum() / (fm * w).sum().clamp_min(1)
            l_uv = A.uv_uv_weight * (F.binary_cross_entropy_with_logits(puv, tgt_uv, reduction='none') * fm).sum() / fm.sum()
            l_n = (F.l1_loss(pN, N, reduction='none') * fm).sum() / fm.sum()
            l_asr = (F.l1_loss(pasr, ten, reduction='none') * tm[:, None]).sum() / tm.sum() / 512
            return [l_dur, l_f0, l_n, l_asr, l_uv]

        NLOSS = 5
    else:
        model = TextStudent('s+mlp').to(DEV)
        model.load_state_dict(torch.load(A.init, map_location=DEV, weights_only=False)['model'])

        def losses(model, batch):
            ids, L, dur, ten, aln, F0, N, tm, fm = batch
            pd_, pF0, pN, pasr = model(ids, L, aln)
            l_dur = (F.l1_loss(torch.log(pd_.clamp_min(1e-3)), torch.log(dur.clamp_min(1e-3)), reduction='none') * tm).sum() / tm.sum()
            l_f0 = (F.l1_loss(pF0, F0, reduction='none') * fm).sum() / fm.sum() / 100
            l_n = (F.l1_loss(pN, N, reduction='none') * fm).sum() / fm.sum()
            l_asr = (F.l1_loss(pasr, ten, reduction='none') * tm[:, None]).sum() / tm.sum() / 512
            return [l_dur, l_f0, l_n, l_asr]

        NLOSS = 4

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
            pd_ = model(ids, L)[0]   # forward returns (dur, lf0n, uv, N, asr) in UV mode
            p = torch.round(pd_[0]).clamp(min=1).cpu()
            mae.append((p - r['pred_dur'].float()).abs().mean().item())
            ratio.append(float(p.sum() / r['pred_dur'].sum()))
        model.train()
        d = dict(l_dur=ls[0], l_f0=ls[1], l_n=ls[2], l_asr=ls[3], dur_mae_frames=sum(mae) / len(mae),
                 length_ratio=sum(ratio) / len(ratio))
        if UV:
            d['l_uv'] = ls[4]
        return d

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
            if UV:
                sc_ = ev_['l_dur'] + ev_['l_asr'] + ev_['l_uv']   # REVIEW Phase 1: selection includes V/UV loss
            else:
                sc_ = ev_['l_dur'] + ev_['l_asr'] if A.score == 'dur_asr' else ev_['l_dur'] + ev_['l_f0'] + ev_['l_n'] + ev_['l_asr']
            print('EVAL', step, json.dumps({k: round(v, 3) for k, v in ev_.items()}), 'score', round(sc_, 3), flush=True)
            if sc_ < best[0]:
                best[:] = [sc_, step, {k: v.detach().clone() for k, v in model.state_dict().items()}]
            elif step - best[1] >= A.patience * A.eval_every:
                print('early stop, best step', best[1], flush=True)
                break
        if step % 100 == 0:
            uvmsg = f' uv {ls[4].item():.3f}' if UV else ''
            print(f'step {step} loss {loss.item():.4f} dur {ls[0].item():.3f} f0 {ls[1].item():.3f} n {ls[2].item():.3f} asr {ls[3].item():.3f}{uvmsg} {(time.time() - T0) / step:.2f}s/step', flush=True)
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


def full(r, gta=False):
    aln = alignment(r['pred_dur'])
    Fr = aln.shape[1]
    if gta and 'gta_F0' in r:   # REVIEW Phase 2: train on the text model's own predicted inputs
        return r['gta_asr'][:, :Fr], r['gta_F0'][:2 * Fr], r['gta_N'][:2 * Fr], r['audio'][:600 * Fr].float() / 32767
    asr = r['t_en'].float() @ aln
    return asr, r['F0'][:2 * Fr], r['N'][:2 * Fr], r['audio'][:600 * Fr].float() / 32767


def segment(r, seg):
    gta = A.gta_p > 0 and 'gta_F0' in r and random.random() < A.gta_p
    asr, F0, N, au = full(r, gta)
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
    c0 = (F0[:2 * a].double() * 300 / 24000).sum() % 1   # phase from the INPUT F0 (GTA when gta)
    w = 0.0 if gta else 1.0   # GTA items: harm/mod losses off (their pitch no longer matches the target audio)
    return asr[:, a:a + seg], F0[2 * a:2 * (a + seg)], N[2 * a:2 * (a + seg)], au[600 * a:600 * (a + seg)], c0.float(), w


def collate_dec(batch, seg):
    x = [segment(r, seg) for r in batch]
    out = [torch.stack([t[i] for t in x]).to(DEV) for i in range(5)]
    out.append(torch.tensor([t[5] for t in x], device=DEV))
    return out


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


def harm_loss(y, t, f0, weights=None, n=2048, hop=512):
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
    per = ((out[0] - out[1]).abs() * v[:, None]).sum((1, 2)) / (v.sum(1) * 8).clamp_min(1)
    return per.mean() if weights is None else (per * weights).sum() / weights.sum().clamp_min(1)


MODB = [(4, 8), (8, 12), (12, 16), (16, 21), (21, 29)]


def mod_loss(y, t, cyc, uv, weights=None, n=64, hop=16, win=75):
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
    per = ((D[0] - D[1]).abs() * vm).sum((1, 2, 3)) / (vm.sum((1, 2, 3)) * 20).clamp_min(1)
    return per.mean() if weights is None else (per * weights).sum() / weights.sum().clamp_min(1)


model = VDecoder(head='v2', dim=160, blocks=4).to(DEV)
model.load_ckpt(A.init, DEV)
if A.gan:
    from discriminators import Discriminators, d_loss, g_loss
    D = Discriminators().to(DEV)
    optD = torch.optim.AdamW(D.parameters(), lr=A.dlr, betas=(0.8, 0.99), weight_decay=A.wd)
opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.8, 0.99), weight_decay=A.wd)
AMP = A.gan and DEV == 'cuda'   # REVIEW Phase 3: fp16 ONLY for the generator forward, not the GAN losses
scaler = torch.cuda.amp.GradScaler(enabled=AMP)
nan_count = 0


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


@torch.no_grad()
def render_gate(stepn):
    """Render the first held sentences at a checkpoint step for quick listening."""
    model.eval()
    d = os.path.join(A.out, f'gate_{stepn}')
    os.makedirs(d, exist_ok=True)
    for i, r in enumerate(held[:12]):
        asr, F0, N, au = full(r)
        torch.manual_seed(i)
        y = model(asr[None].to(DEV), F0[None].to(DEV), N[None].to(DEV)).squeeze().cpu()
        sf.write(os.path.join(d, f'{i + 1}.wav'), y.numpy(), 24000)
    model.train()


step = 0
model.train()
before = evaluate()
print('before:', json.dumps(before), flush=True)
while step < A.steps and (time.time() - T0) / 60 < A.max_minutes:
    asr, F0, N, au, c0, wg = collate_dec(random.sample(train, min(A.bs, len(train))) if len(train) >= A.bs else [random.choice(train) for _ in range(A.bs)], A.seg)
    for g in opt.param_groups:
        g['lr'] = LR * cosine(step, A.steps)
    if A.aug > 0:
        F0_in = F0 * torch.exp(jitter(F0, 0.08 * A.aug)) * (F0 > 0)
        N_in = N + jitter(N, 0.4 * A.aug)
    else:
        F0_in, N_in = F0, N
    warm = A.gan and step < A.gan_warmup
    adv = fm = ld = None
    with torch.autocast(device_type=DEV, dtype=torch.float16, enabled=AMP):
        y = model(asr, F0_in, N_in, c0=c0)
    y = y.float()   # GAN losses (squared LSGAN terms) MUST be fp32 - they overflow in fp16
    lm, ls = spec_losses(y, au)
    lh = harm_loss(y, au, F0, weights=wg)
    lmod = mod_loss(y, au, model.head.last['cyc'], model.head.last['uv'], weights=wg)
    if A.gan:
        ld = d_loss(D, au, y.detach())
        if not warm:
            for p in D.parameters():
                p.requires_grad_(False)
            adv, fm = g_loss(D, au, y)
            for p in D.parameters():
                p.requires_grad_(True)
            loss = A.mel_weight * (lm + ls) + adv + 2.0 * fm + 2.0 * lh + 5.0 * lmod
        else:
            loss = torch.zeros((), device=DEV)   # D warm-up: G is frozen
    else:
        loss = lm + ls + 2.0 * lh + 5.0 * lmod
    bad = A.gan and (not torch.isfinite(ld) or not torch.isfinite(loss))
    if bad:
        nan_count += 1
        print(f'NONFINITE step {step} d={ld.item()} g={loss.item()} count={nan_count}', flush=True)
        if nan_count > 20:
            raise SystemExit('GAN diverged: too many non-finite steps')
    if A.gan and not bad:
        optD.zero_grad()
        scaler.scale(ld).backward()
        scaler.unscale_(optD)
        torch.nn.utils.clip_grad_norm_(D.parameters(), 5.0)
        scaler.step(optD)
        if not warm:
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
        scaler.update()
    elif not A.gan:
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
    step += 1
    if step % 1000 == 0:
        torch.save({'model': model.state_dict(), 'step': step}, os.path.join(A.out, f'step{step}.pt'))
    if A.gan and step in (2000, 4000, 8000):
        render_gate(step)
    if step % 100 == 0 or step == 50:
        gmsg = '' if adv is None else f' adv {adv.item():.3f} fm {fm.item():.3f} d {ld.item():.3f}'
        print(f'step {step} loss {loss.item():.4f} mel {lm.item():.3f} stft {ls.item():.3f} harm {lh.item():.3f} mod {lmod.item():.3f}{gmsg}' + (' WARM' if warm else '') + f' {(time.time() - T0) / step:.2f}s/step', flush=True)
    if step % 500 == 0:
        print('EVAL', json.dumps(evaluate()), flush=True)
after = evaluate(save=True)
after.update(step=step, minutes=(time.time() - T0) / 60, before=before['held_spec_loss'])
print('after:', json.dumps(after), flush=True)
log.write(json.dumps(after) + '\n')
torch.save({'model': model.state_dict(), 'step': step}, os.path.join(A.out, 'last.pt'))
