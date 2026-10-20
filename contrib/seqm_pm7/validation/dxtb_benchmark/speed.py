"""dxtb の速度：エネルギー＋力（自動微分）。各条件で 1 回目（初期化込み）と 2 回目を測る"""
import os, sys, json, time, itertools, numpy as np, ase.io, torch
sys.path.insert(0, ".")
from ase import Atoms
from common import *
W = os.path.abspath("../..")
def load(nm): a = ase.io.read(f"{W}/pm7d/mols2/{nm}/xtbopt.xyz"); return a.get_chemical_symbols(), a.positions - a.positions.mean(0)
def grid(s, x, n, d): offs = [np.array(v) * d for v in itertools.product(range(n), repeat=3)]; return s * len(offs), np.vstack([x + o for o in offs])
R = json.load(open(f"{W}/seqm_pm7/validation/mopac_ref.json")); w = R["cl:HF_w20_start_neutral"]
s1, x1 = load("TDMAZ"); s2, x2 = load("Zr4O6OH4")
sets = {"TDMAZ (37)": (s1, x1),
        "4x Zr4O6(OH)4 (72)": (s2 * 4, np.vstack([x2 + np.array(v) for v in ([0,0,0],[9,0,0],[0,9,0],[9,9,0])])),
        "4x TDMAZ (148)": (s1 * 4, np.vstack([x1 + np.array(v) for v in ([0,0,0],[11,0,0],[0,11,0],[11,11,0])])),
        "27x Zr4O6(OH)4 (486)": grid(s2, x2, 3, 9.0), "27x TDMAZ (999)": grid(s1, x1, 3, 11.0),
        "27x HF(H2O)20 (1674)": grid(w["sym"], np.array(w["x"]), 3, 12.0)}
conds = sys.argv[1].split(",")       # 例: cpu64,cuda64,cuda32,cuda64b8
only = sys.argv[2].split(",") if len(sys.argv) > 2 else None
torch.set_num_threads(4)
fn = "dxtb_speed.json"; out = json.load(open(fn)) if os.path.exists(fn) else []
rng = np.random.default_rng(0)
for nm, (s, x) in sets.items():
    if only and not any(o in nm for o in only): continue
    a = Atoms(s, positions=x)
    for c in conds:
        dev = "cuda:0" if c.startswith("cuda") else "cpu"; dt = torch.float32 if "32" in c else torch.float64
        nb = 8 if c.endswith("b8") else 1
        # まとめ計算：わずかに乱した 8 構造（全部同じだと意味がないため）
        L = [Atoms(s, positions=x + (rng.normal(0, 0.02, x.shape) if i else 0)) for i in range(nb)]
        rec = dict(system=nm, natoms=len(s), cond=c, batch=nb)
        try:
            for rep in (0, 1):
                if dev != "cpu": torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                t = time.time(); E, F = ef(L, 2, dev, dt)
                if dev != "cpu": torch.cuda.synchronize()
                rec[f"t{rep}"] = time.time() - t
            rec["E0_eV"] = float(E[0]); rec["per_struct_s"] = rec["t1"] / nb
            if dev != "cpu": rec["peak_GB"] = torch.cuda.max_memory_allocated() / 1e9
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {str(e)[:120]}"
        if dev != "cpu": torch.cuda.empty_cache()
        out = [r for r in out if not (r["system"] == nm and r["cond"] == c)] + [rec]
        json.dump(out, open(fn, "w"), indent=1)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.items() if k != "E0_eV"}, flush=True)
