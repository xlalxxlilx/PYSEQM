import json, os, re, subprocess, glob, numpy as np, ase.io, sys
sys.path.insert(0, ".")
from common import *
XTB = os.environ.get("XTB_BIN", "xtb")
X = json.load(open("../rxn/xtb.json")); out = {"species": {}, "path": {}}
def xtb_grad(a, g, d):
    os.makedirs(d, exist_ok=True); ase.io.write(f"{d}/m.xyz", a)
    subprocess.run([XTB, "m.xyz", "--gfn", str(g), "--grad", "--acc", "0.1"], cwd=d, capture_output=True, env={**os.environ, "OMP_NUM_THREADS": "4"})
    L = open(f"{d}/gradient").read().split("\n"); n = len(a)
    G = np.array([[float(v.replace("D", "E")) for v in l.split()] for l in L[2 + n:2 + 2 * n]])
    return -G * EH2EV * AA2AU
for f in sorted(glob.glob("../rxn/geo/*.xyz")):
    k = os.path.basename(f)[:-4]; a = ase.io.read(f)
    for g in (2, 1):
        E, F = ef([a], g); Fx = xtb_grad(a, g, f"w/{k}_{g}")
        out["species"].setdefault(k, {})[f"GFN{g}"] = dict(E_dxtb=float(E[0]), E_xtb=X[k][f"GFN{g}"], dF_max=float(np.abs(F[0] - Fx).max()), F_max=float(np.abs(Fx).max()))
    print(k, {m: (round(v["E_dxtb"] - v["E_xtb"], 6), round(v["dF_max"], 6)) for m, v in out["species"][k].items()}, flush=True)
W = os.path.abspath("../../pyseqm_test/scans"); P = json.load(open("../path/xtb_on_path.json"))
for tag in ("HF_w4", "HF_w8", "NH3_w4", "NH3_w8"):
    fr = ase.io.read(f"{W}/{tag}_UMA_scan.xyz", index=":") + [ase.io.read(f"{W}/{tag}_UMA_molopt.xyz"), ase.io.read(f"{W}/{tag}_UMA_release.xyz")]
    for g in (2, 1):
        E, _ = ef(fr, g)          # 21 構造をまとめて計算
        ref = np.array(P[tag][f"GFN{g}"]); out["path"].setdefault(tag, {})[f"GFN{g}"] = dict(E_dxtb=E.tolist(), maxdiff_eV=float(np.abs(E - ref).max()),
            maxdiff_rel_kJ=float(np.abs((E - E[19]) - (ref - ref[19])).max() * 96.485332))
    print(tag, {m: (v["maxdiff_eV"], v["maxdiff_rel_kJ"]) for m, v in out["path"][tag].items()}, flush=True)
json.dump(out, open(os.environ.get("ACC_OUT", "dxtb_accuracy.json"), "w"), indent=1)
