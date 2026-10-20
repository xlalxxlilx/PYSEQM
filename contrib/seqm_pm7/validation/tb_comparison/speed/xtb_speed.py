import os, json, time, itertools, subprocess, re, numpy as np, ase.io
from ase import Atoms
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
out = []
for nm, (s, x) in sets.items():
    d = "run_" + nm.split("(")[1].rstrip(")"); os.makedirs(d, exist_ok=True); ase.io.write(f"{d}/m.xyz", Atoms(s, positions=x))
    for th in (4, 8):
        t = time.time()
        r = subprocess.run(["xtb", "m.xyz", "--gfn", "2", "--grad"], cwd=d, capture_output=True, text=True,
                           env={**os.environ, "OMP_NUM_THREADS": str(th), "MKL_NUM_THREADS": str(th), "OMP_STACKSIZE": "2G"})
        dt = time.time() - t
        cyc = re.findall(r"converged in\s+(\d+)\s+iterations", r.stdout) or re.findall(r"SCC converged in\s+(\d+)", r.stdout)
        ok = "TOTAL ENERGY" in r.stdout
        out.append(dict(system=nm, natoms=len(s), threads=th, t=dt, ok=ok))
        print(f"{nm:22s} GFN2 {th} thr {dt:7.1f}s ok {ok}", flush=True)
        json.dump(out, open("xtb_speed.json", "w"), indent=1)
