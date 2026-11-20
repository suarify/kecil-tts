from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
logs = api.kernels_logs('suarifymalaysia/sm-29-nicole-train')
txt = logs if isinstance(logs, str) else str(logs)
open('runs/sm29_log.txt', 'w', encoding='utf-8', errors='replace').write(txt)
print('LEN', len(txt))
print(txt[-5000:])
