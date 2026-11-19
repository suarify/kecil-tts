import subprocess

msg = """
## 2026-11-20 Space UI v6 (5 owner requests) - uploaded + committed to GitHub
- (1) Nicole speaker sub no longer says "af_nicole"; note simplified.
- (2) "Runs anywhere" modal title no longer prefixed with "3 ·" (EN + BM).
- (3) The two modal SVGs now gently float/rotate via @keyframes floaty (3.2s/3.8s, staggered), disabled under prefers-reduced-motion.
- (4) Removed the old Yati note "Understandable and the most accurate so far...".
- (5) Footer gets a header-like gradient separator: 3px teal->navy->teal bar on top, 4px navy->teal bar at bottom, subtle tint background.
- Verified live via browser: 3 speakers, Nicole sub "female voice", modal title clean, svgAnim "floaty", footer gradient present, zero JS errors.
"""
with open('STATUS.md', 'a', encoding='utf-8') as f:
    f.write(msg)
files = ['STATUS.md', 'scripts/polish_space6.py', 'scripts/verify_space6.py', 'scripts/inspect_ui6.py']
subprocess.run(['git', 'add'] + files, check=True)
r = subprocess.run(['git', 'commit', '-m', 'Space UI v6: Nicole label, modal title, animated SVG, footer gradient', '--quiet'], capture_output=True, text=True)
print('commit rc', r.returncode, '| status:', subprocess.run(['git','status','--short'],capture_output=True,text=True).stdout.strip() or 'clean')
