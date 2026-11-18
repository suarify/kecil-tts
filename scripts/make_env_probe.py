"""Probe whether a notebook kernel with language_info 3.11 actually gets Python 3.11."""
import json

code = """import sys, glob, subprocess
print('PYVER', sys.version)
print('INPUTS', glob.glob('/kaggle/input/*'))
p = subprocess.run(['pip', 'show', 'torch', 'torchaudio', 'transformers'], capture_output=True, text=True)
print(p.stdout[:3000])
"""
nb = {
    "cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code.splitlines(True)}],
    "metadata": {
        "kernelspec": {"display_name": "Python 3.11", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
import os
os.makedirs('kaggle/sm17_env_probe', exist_ok=True)
json.dump(nb, open('kaggle/sm17_env_probe/sm17_env_probe.ipynb', 'w', encoding='utf-8'), indent=1)
meta = {
    "id": "suarifymalaysia/sm-17-env-probe", "title": "sm 17 env probe",
    "code_file": "sm17_env_probe.ipynb", "language": "python", "kernel_type": "notebook",
    "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
    "keywords": ["gpu"], "dataset_sources": ["suarifymalaysia/sm12-in"], "kernel_sources": [],
    "competition_sources": [], "model_sources": [], "machine_shape": "NvidiaTeslaT4",
}
json.dump(meta, open('kaggle/sm17_env_probe/kernel-metadata.json', 'w', encoding='utf-8'), indent=1)
print('probe notebook written')
