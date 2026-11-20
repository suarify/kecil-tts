import sys
from kaggle.api.kaggle_api_extended import KaggleApi
sys.stdin = open('runs/yes.txt', 'w+')
sys.stdin.write('yes\n'); sys.stdin.seek(0)
api = KaggleApi(); api.authenticate()
try:
    api.kernels_delete('suarifymalaysia/username-probe-delete-me')
    print('deleted probe kernel')
except Exception as e:
    print('err', type(e).__name__, str(e)[:120])
