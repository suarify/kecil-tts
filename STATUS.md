# STATUS (continue.plan.md)

## 2026-11-14 Phase A, search done, waiting on Gate A (owner listens)
- Kernel `kmb-04-teacher-search` v1 ran about 51 min on a T4. Accent-score check: 100% (voice-converted Chatterbox clip closer to the Chatterbox reference than stock Bella on every sentence).
- Stock Bella, 16 search sentences: CER 6.0%, accent distance 0.2345.
- Best blends keep Bella's timbre only partly and take prosody entirely from the donor (wp = 0): ef_dora a0.5 (accent 0.161, CER 0.2%), pf_dora a0.5 (0.163), pf_dora a0.7 (0.164, tap r), ef_dora a0.3 (0.164). Prosody blended 0.5 and Chinese/French/Hindi/Japanese donors were worse.
- Finalists F1..F4 on 32 ref + 3 number sentences: CER F1 2.1%, F2 2.7%, F3 2.6%, F4 2.2% vs stock Bella 3.7%. Test_40 CER: F1 2.9%, F2 4.5%, F3 4.5%, F4 3.5%.
- final_a='ə' (saya -> sajə) made almost no automatic difference on F1 (CER 2.04% vs 2.14%, accent 0.1674 vs 0.1657). Included as F1_alt for the owner.
- Page: runs/listen/teacher.html. Gate A pending.
