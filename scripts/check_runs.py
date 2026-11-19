from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
s = api.kernels_status('suarifymalaysia/sm-21-gan')
print('sm-21-gan STATUS', getattr(s.status, 'name', None))
try:
    ks = api.kernels_list(user='suarifymalaysia', page_size=8)
    for k in ks:
        print(getattr(k, 'ref', k), '|', getattr(k, 'title', ''))
except Exception as e:
    print('list err', type(e).__name__, str(e)[:150])
