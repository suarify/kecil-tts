import configparser
import os

from huggingface_hub import HfApi

cp = configparser.ConfigParser()
cp.read(os.path.expanduser('~/.cache/huggingface/stored_tokens'))
api = HfApi(token=cp['Suarify Malaysia']['hf_token'])
space = 'Suarify/suaradee-web'
try:
    logs = api.get_space_runtime_logs(repo_id=space)
    print(str(logs)[-4000:])
except Exception as e:
    print('LOGS ERR', type(e).__name__, str(e)[:300])
try:
    rt = api.get_space_runtime(repo_id=space)
    print('RUNTIME', rt)
except Exception as e:
    print('RUNTIME ERR', type(e).__name__, str(e)[:300])
