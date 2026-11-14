# STATUS (continue.plan.md)

## 2026-11-14 Phase A, search done, waiting on Gate A (owner listens)
- Kernel `kmb-04-teacher-search` v1 ran about 51 min on a T4. Accent-score check: 100% (voice-converted Chatterbox clip closer to the Chatterbox reference than stock Bella on every sentence).
- Stock Bella, 16 search sentences: CER 6.0%, accent distance 0.2345.
- Best blends keep Bella's timbre only partly and take prosody entirely from the donor (wp = 0): ef_dora a0.5 (accent 0.161, CER 0.2%), pf_dora a0.5 (0.163), pf_dora a0.7 (0.164, tap r), ef_dora a0.3 (0.164). Prosody blended 0.5 and Chinese/French/Hindi/Japanese donors were worse.
- Finalists F1..F4 on 32 ref + 3 number sentences: CER F1 2.1%, F2 2.7%, F3 2.6%, F4 2.2% vs stock Bella 3.7%. Test_40 CER: F1 2.9%, F2 4.5%, F3 4.5%, F4 3.5%.
- final_a='ə' (saya -> sajə) made almost no automatic difference on F1 (CER 2.04% vs 2.14%, accent 0.1674 vs 0.1657). Included as F1_alt for the owner.
- Page: runs/listen/teacher.html. Gate A pending.

## 2026-11-14 Gate A result: FAILED (owner ears)
- Owner picks, 15 of 16 rated: Chatterbox voice-converted clip 12, stock Bella 2 (both on number sentences that had no Chatterbox clip), F1 1. Not acceptable: F2 3, F1 2, F3 2, F4 2, F1_alt 2, B0 3.
- Owner: Dora-based voices "not so accurate" on the last few sentences (code-switch and numbers).
- Findings: n10 (RM87.20) was a NORMALISER BUG: the digit-context rule ("akaun ... RM87") ate the digits, so every system read "RM lapan tujuh". Fixed in Python and JS, test added, parity 2000/2000, commit 60b83c5. n05 (OTP) read "kodotipi": the donor prosody runs the letters together.
- Per section 12: skipped step 1 (widening the search; the automatic accent score already saturated at 0.161-0.177 while the owner still picked Chatterbox every time it was present). Running step 3 proof only (50-clip scale, here 32 sentences): kernel kmb-05-transplant = F1 + Chatterbox word timings (TPd) + Chatterbox pitch (TPf). Page: runs/listen/transplant.html.
