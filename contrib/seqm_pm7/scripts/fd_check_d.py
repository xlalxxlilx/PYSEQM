"""力の検算: 自前の力 vs 自前のエネルギーの中心差分、MOPAC の勾配 vs MOPAC のエネルギーの中心差分"""
import json, os, sys, re, subprocess, tempfile, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import seqm_pm7
REF = json.load(open(os.path.join(os.path.dirname(__file__), "..", "validation", "mopac_ref_d.json")))
MOP = os.path.expanduser(os.environ.get("MOPAC_HOME", "~/opt/mopac-23.2.5-linux") + "/bin/mopac"); LIB = os.path.expanduser(os.environ.get("MOPAC_HOME", "~/opt/mopac-23.2.5-linux") + "/lib")
def mopac_H(s, x, kw):
    d = tempfile.mkdtemp(prefix="mopf_")
    open(f"{d}/m.mop", "w").write(f"{kw} 1SCF PRECISE AUX(PRECISION=9)\n\n\n" + "".join(f"{a} {c[0]:.10f} 1 {c[1]:.10f} 1 {c[2]:.10f} 1\n" for a, c in zip(s, x)))
    subprocess.run([MOP, "m.mop"], cwd=d, capture_output=True, env={**os.environ, "LD_LIBRARY_PATH": LIB, "OMP_NUM_THREADS": "1"})
    return float(re.search(r"HEAT_OF_FORMATION:KCAL/MOL=([-+0-9.EeD]+)", open(f"{d}/m.aux").read()).group(1).replace("D", "E"))
for nm, m in (("SiH2NMe22", "PM6"), ("TDMAZ", "PM6"), ("Al2O3H2", "PM7")):
    r = REF[nm]; s, x = r["sym"], np.array(r["x"]); q = seqm_pm7.compute(s, x, m); G = np.array(r[m]["grad"])
    k = np.unravel_index(np.argmax(np.abs(-q["forces"] - G)), G.shape); h = 1e-3
    xp = x.copy(); xp[k] += h; xm = x.copy(); xm[k] -= h
    fd_ours = (seqm_pm7.compute(s, xp, m)["Hf"] - seqm_pm7.compute(s, xm, m)["Hf"]) / (2 * h)
    fd_mop = (mopac_H(s, xp, m) - mopac_H(s, xm, m)) / (2 * h)
    print(f"{nm} {m} 成分 {k}: 自前の勾配 {-q['forces'][k]:+.4f}, 自前 FD {fd_ours:+.4f} | MOPAC 勾配 {G[k]:+.4f}, MOPAC FD {fd_mop:+.4f}")
