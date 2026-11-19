import json
r = json.load(open('runs/kaggle_sm25/sm25_report.json'))
for cat in ('numbers', 'codeswitch'):
    print('===', cat)
    for x in [x for x in r['rows'] if x['cat'] == cat][:6]:
        print('REF:', x['ref'][:120])
        print('HYP:', x['hyp'][:120], '| CER', x['cer'])
        print()
