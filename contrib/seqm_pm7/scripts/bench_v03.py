"""seqm_pm7 v0.2 相当（旧）と v0.3（新）の速度比較。PM7、単点計算 + 力。
使い方: python bench_v03.py <cpu|cuda> <batch>
"""
import sys, os, time, json, numpy as np, ase.io, torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))); torch.set_num_threads(4)
import seqm_pm7.calculator as C, seqm_pm7.fock_fast as FF, seqm_pm7.rot_fast as RF
import seqm.seqm_functions.two_elec_two_center_int as TT
D = os.environ.get("SEQM_PM7_MOLS", os.path.join(os.path.dirname(__file__), "..", "validation", "mols2")) + "/"
def load(nm): a = ase.io.read(f"{D}{nm}/xtbopt.xyz"); return a.get_chemical_symbols(), a.positions - a.positions.mean(0)
s1, x1 = load("TDMAZ"); s2, x2 = load("Zr4O6OH4")
sets = {"TDMAZ (37)": (s1, x1),
        "4x Zr4O6(OH)4 (72)": (s2 * 4, np.vstack([x2 + np.array(v) for v in ([0,0,0],[9,0,0],[0,9,0],[9,9,0])])),
        "4x TDMAZ (148)": (s1 * 4, np.vstack([x1 + np.array(v) for v in ([0,0,0],[11,0,0],[0,11,0],[11,11,0])]))}
def mode(new):
    FF.ENABLED = new
    TT.GenerateRotationMatrix = RF.GenerateRotationMatrix_fast if new else RF._ORIG
    return dict(pre_iter=8, pre_mode="inline") if new else dict(pre_iter=25, pre_mode="separate")
dev = sys.argv[1]; nb = int(sys.argv[2]); out = []
for nm, (s, x) in sets.items():
    X = np.stack([x + np.random.default_rng(k).normal(0, 0.01, x.shape) for k in range(nb)]) if nb > 1 else x
    res = {}
    for new in (False, True):
        kw = mode(new); C.compute(s, X, "PM7", device=dev, **kw)
        if dev == "cuda": torch.cuda.synchronize()
        t = time.time(); q = C.compute(s, X, "PM7", device=dev, **kw)
        if dev == "cuda": torch.cuda.synchronize()
        res[new] = (time.time() - t, q)
    dH = float(np.max(np.abs(np.asarray(res[True][1]["Hf"]) - np.asarray(res[False][1]["Hf"]))))
    dF = float(np.max(np.abs(res[True][1]["forces"] - res[False][1]["forces"])))
    r = dict(device=dev, batch=nb, system=nm, old_s=res[False][0], new_s=res[True][0], speedup=res[False][0] / res[True][0], max_dHf=dH, max_dF=dF)
    out.append(r)
    print(f"{dev} b{nb} {nm:20s} old {r['old_s']:6.2f}s  new {r['new_s']:6.2f}s  x{r['speedup']:.2f}  per-struct new {r['new_s']/nb:.2f}s  |dHf| {dH:.1e} |dF| {dF:.1e}", flush=True)
json.dump(out, open(f"bench_v03_{dev}_b{nb}.json", "w"), indent=1)
