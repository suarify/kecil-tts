"""Text side with log-F0 regression + explicit voiced/unvoiced classifier.

Replaces the old 'F0 in Hz with 0 for unvoiced' target that forced the regressor to hedge at
voicing edges (17% dropouts, 25% V/UV errors in the v0.4.1 baseline). Keeps the old TextStudent
path working: the new class subclasses it and only overrides the prosody prediction.

Init from the v0.4.1 text checkpoint with strict=False (uv_proj is new); F0_proj bias is reset to 0
because it now predicts normalised log-F0, not Hz.
"""
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from student import TextStudent  # noqa: E402


class TextStudentUV(TextStudent):
    def __init__(self, preset="s+mlp", n_token=178, teacher_dim=512, max_dur=50, n_voices=1,
                 mu=5.2, sd=0.5):
        super().__init__(preset=preset, n_token=n_token, teacher_dim=teacher_dim,
                         max_dur=max_dur, n_voices=n_voices)
        C = self.predictor.F0_proj.in_channels
        self.uv_proj = nn.Conv1d(C, 1, 1)
        nn.init.constant_(self.uv_proj.bias, math.log(0.6 / 0.4))  # logit(voiced_fraction ~0.6)
        nn.init.normal_(self.uv_proj.weight, mean=0.0, std=0.001)
        # The old F0_proj was trained to emit Hz (~200); it must start near 0 for normalised log-F0.
        nn.init.normal_(self.predictor.F0_proj.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.predictor.F0_proj.bias)
        self.register_buffer('lf0_mu', torch.tensor(float(mu)))
        self.register_buffer('lf0_sd', torch.tensor(float(sd)))

    def prosody(self, en, s):
        """en [B, F, d_hid], s [B, style] -> lf0n, uv_logit, N all [B, F']."""
        x, _ = self.predictor.shared(en.transpose(-1, -2))     # [B, d_hid//2, F']
        h = x.transpose(-1, -2)                                # [B, F', d_hid//2]
        z = h
        for block in self.predictor.F0:
            z = block(z, s)
        lf0n = self.predictor.F0_proj(z).squeeze(1)            # [B, F']
        uv = self.uv_proj(z).squeeze(1)                        # [B, F']
        z = h
        for block in self.predictor.N:
            z = block(z, s)
        N = self.predictor.N_proj(z).squeeze(1)                # [B, F']
        return lf0n, uv, N

    def forward(self, ids, lengths, aln=None, voice=None, **kw):
        """Same interface as TextStudent but returns (dur, lf0n, uv_logit, N, asr_tok)."""
        B, T = ids.shape
        mask = torch.arange(T, device=ids.device)[None].expand(B, -1) >= lengths[:, None]
        s = self.style[voice] if voice is not None else self.style[:1].expand(B, -1)
        bert = self.bert(ids, attention_mask=(~mask).int())
        d_en = self.bert_encoder(bert).transpose(-1, -2)
        d = self.predictor.text_encoder(d_en, s, lengths, mask)
        x, _ = self.predictor.lstm(d)
        dur = torch.sigmoid(self.predictor.duration_proj(x)).sum(-1)
        t_en = self.text_encoder(ids, lengths, mask)
        asr_tok = self.asr_proj(t_en.transpose(-1, -2)).transpose(-1, -2)
        if aln is None:
            return dur, None, None, None, asr_tok
        en = d.transpose(-1, -2) @ aln
        lf0n, uv, N = self.prosody(en, s)
        return dur, lf0n, uv, N, asr_tok

    def f0_hz(self, lf0n, uv_logit):
        """lf0n, uv_logit [B,F] -> F0 in Hz [B,F] with 0 where unvoiced; majority-filter the mask."""
        m = (torch.sigmoid(uv_logit) > 0.5).float().unsqueeze(1)   # [B,1,F] for avg_pool1d
        pad = m.shape[-1] % 2
        if pad:
            m = F.pad(m, (0, 1))
        m = (F.avg_pool1d(m, 5, 1, 2, count_include_pad=False) > 0.5).squeeze(1)
        f0 = torch.exp(lf0n * self.lf0_sd + self.lf0_mu) * m.float()
        return f0[..., :lf0n.shape[-1]]


def reinit_f0_proj(model):
    """Call AFTER loading the v0.4.1 checkpoint: the old F0_proj emitted Hz (~200),
    the new one must start near 0 for normalised log-F0 targets."""
    nn.init.normal_(model.predictor.F0_proj.weight, mean=0.0, std=0.01)
    nn.init.zeros_(model.predictor.F0_proj.bias)


def set_lf0_stats(model, rows):
    """Compute mu/sd of log-F0 over voiced frames of the training rows."""
    vals = []
    for r in rows:
        f = r['F0'].numpy() if hasattr(r['F0'], 'numpy') else r['F0']
        v = f[f > 50]
        if len(v):
            vals.append(np.log(v))
    x = np.concatenate(vals)
    model.lf0_mu.fill_(float(x.mean()))
    model.lf0_sd.fill_(float(x.std()))
    return float(x.mean()), float(x.std())
