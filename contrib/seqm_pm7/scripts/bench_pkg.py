"""指定したパッケージの seqm_pm7 で PM7（単点 + 力）を測る。使い方: python bench_pkg.py <pkgdir> <cpu|cuda> <batch>"""
import sys, os, time, json, numpy as np, ase.io, torch
pkg, dev, nb = sys.argv[1], sys.argv[2], int(sys.argv[3])
sys.path.insert(0, os.path.abspath(pkg)); torch.set_num_threads(4)
import seqm_pm7, seqm_pm7.calculator as C
D = os.environ.get("SEQM_PM7_MOLS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "validation", "mols2")) + "/"
def load(nm): a = ase.io.read(f"{D}{nm}/xtbopt.xyz"); return a.get_chemical_symbols(), a.positions - a.positions.mean(0)
s1, x1 = load("TDMAZ"); s2, x2 = load("Zr4O6OH4")
sets = {"TDMAZ (37)": (s1, x1),
        "4x Zr4O6(OH)4 (72)": (s2 * 4, np.vstack([x2 + np.array(v) for v in ([0,0,0],[9,0,0],[0,9,0],[9,9,0])])),
        "4x TDMAZ (148)": (s1 * 4, np.vstack([x1 + np.array(v) for v in ([0,0,0],[11,0,0],[0,11,0],[11,11,0])]))}
out = []
for nm, (s, x) in sets.items():
    X = np.stack([x + np.random.default_rng(k).normal(0, 0.01, x.shape) for k in range(nb)]) if nb > 1 else x
    C.compute(s, X, "PM7", device=dev)
    if dev == "cuda": torch.cuda.synchronize()
    ts = []
    for _ in range(2):
        t = time.time(); q = C.compute(s, X, "PM7", device=dev)
        if dev == "cuda": torch.cuda.synchronize()
        ts.append(time.time() - t)
    out.append(dict(version=seqm_pm7.__version__, device=dev, batch=nb, system=nm, t=min(ts), Hf=np.asarray(q["Hf"]).tolist(), F=np.asarray(q["forces"]).tolist()))
    print(f"v{seqm_pm7.__version__} {dev} b{nb} {nm:20s} {min(ts):6.2f}s  per-struct {min(ts)/nb:.2f}s", flush=True)
json.dump(out, open(f"bench_{seqm_pm7.__version__}_{dev}_b{nb}.json", "w"))
