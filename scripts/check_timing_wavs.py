import glob, os, soundfile as sf
for c in ('A_baseline', 'B_cond_once', 'C_cond_top9'):
    fs = sorted(glob.glob(f'runs/kaggle_sm23/p4/{c}/*.wav'))
    durs = [len(sf.read(f)[0])/24000 for f in fs]
    print(c, len(fs), 'clips | dur median', round(sorted(durs)[len(durs)//2],2) if durs else None,
          '| min', round(min(durs),2) if durs else None, '| max', round(max(durs),2) if durs else None,
          '| MB', round(sum(os.path.getsize(f) for f in fs)/1e6,1))
