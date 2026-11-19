s = open('runs/local_v041/index.html', encoding='utf-8').read()
checks = {
    '1 no af_nicole': 'af_nicole' not in s,
    '1 nicole sub female': "sub: 'female voice'" in s,
    '2 modal no 3-dot': '3 · Runs' not in s and '3 · Berjalan' not in s,
    '2 modal title': 'Runs anywhere: edge, IoT, kiosk' in s,
    '3 svg animation': '@keyframes floaty' in s and 'animation:floaty' in s,
    '3 reduced motion': 'prefers-reduced-motion' in s,
    '4 yati note removed': 'Understandable and the most accurate' not in s,
    '5 footer gradient before': 'footer::before' in s and 'footer::after' in s,
    'nicole speaker intact': "nicole: { name: 'Nicole'" in s,
    'yati speaker intact': "yati: { name: 'Yati'" in s,
}
for k, v in checks.items():
    print(('PASS' if v else 'FAIL'), k)
print('ALL', all(checks.values()))
