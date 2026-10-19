import time, json, numpy as np, torch, os, subprocess, tempfile
import seqm_pm7
from validate import MOPAC
M = json.load(open("mopac_ref.json"))
def tmop(s, x, kw="PM7"):
    d = tempfile.mkdtemp(); open(f"{d}/m.mop","w").write(f"{kw} 1SCF GRAD\n\n\n"+"".join(f"{a} {c[0]:.6f} 1 {c[1]:.6f} 1 {c[2]:.6f} 1\n" for a,c in zip(s,x)))
    t=time.time(); subprocess.run([MOPAC,"m.mop"],cwd=d,capture_output=True,env={**os.environ,"OMP_NUM_THREADS":"4","LD_LIBRARY_PATH":os.path.expanduser(os.environ.get("MOPAC_HOME", "~/opt/mopac-23.2.5-linux") + "/lib")}); return time.time()-t
for nm in ["s22:Water_dimer", "cl:HF_w8", "cl:HF_w20_start_neutral"]:
    s, x = M[nm]["sym"], np.array(M[nm]["x"])
    seqm_pm7.compute(s, x, "PM7")  # ウォームアップ
    t = time.time(); [seqm_pm7.compute(s, x, "PM7") for _ in range(3)]; tc = (time.time()-t)/3
    X8 = np.stack([x + np.random.default_rng(i).normal(0, 0.01, x.shape) for i in range(8)])
    t = time.time(); seqm_pm7.compute(s, X8, "PM7"); tb = (time.time()-t)/8
    tg = None
    if torch.cuda.is_available():
        seqm_pm7.compute(s, X8, "PM7", device="cuda"); torch.cuda.synchronize()
        t = time.time(); seqm_pm7.compute(s, X8, "PM7", device="cuda"); torch.cuda.synchronize(); tg = (time.time()-t)/8
    tm = np.mean([tmop(s, x) for _ in range(3)])
    print(f"{nm:26s} n={len(s):3d}  seqm_pm7 CPU 1本 {tc:.2f} s | CPU 8本バッチ {tb:.2f} s/本 | GPU 8本バッチ {tg if tg is None else round(tg,2)} s/本 | MOPAC 23.2.5 (4 thr) {tm:.2f} s")
