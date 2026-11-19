import subprocess

msg = """
## 2026-11-19 Phase 4 step 1 (CPU) DONE: 3,400 new sentences selected
- Owner said "publish n go phase4" after the modest v0.5 gain. v0.5 published + Space updated (see prior entry).
- `scripts/make_p4_texts.py`: loaded 7,500 corpus rows, excluded the 1,281 already used, greedy token-bigram coverage (40k bigram space; char bigrams saturated at 254 sentences so switched to token bigrams) -> 2,500 sentences covering 30,121 token bigrams. 300 numbers from the numbers category.
- `scripts/make_p4_codeswitch.py`: corpus had only ~218 unused code-switch sentences, so generated 600 Malay-frame + English-loanword sentences from 20 templates x word lists (REVIEW says "English loanwords in Malay frames, use the lexicon from C3").
- `scripts/merge_p4_texts.py` -> `eval/p4_new_texts_final.tsv`: 2,500 greedy + 600 codeswitch (204 corpus + 396 synthetic) + 300 numbers = 3,400 rows, asserted.
- NEXT (GPU, needs owner yes): 50-clip timing kernel testing batched Chatterbox generation + fewer sampling steps; pick fastest config with CER <=2.5% on those 50; report clips/hour and total GPU-h for owner approval BEFORE generating all 3,400.
"""
with open('STATUS.md', 'a', encoding='utf-8') as f:
    f.write(msg)
files = ['STATUS.md', 'scripts/make_p4_texts.py', 'scripts/make_p4_codeswitch.py', 'scripts/merge_p4_texts.py',
         'eval/p4_new_texts.tsv', 'eval/p4_codeswitch_synth.tsv', 'eval/p4_new_texts_final.tsv',
         'scripts/explore_cs.py', 'scripts/check_en_hits.py']
subprocess.run(['git', 'add'] + files, check=True)
subprocess.run(['git', 'commit', '-m', 'REVIEW Phase 4 step 1: 3,400 new training sentences', '--quiet'])
print('committed')
