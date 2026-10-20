import os, glob, json, time, ase.io, numpy as np, warnings
warnings.filterwarnings("ignore")
from pyscf import gto
from gpu4pyscf.dft import rks
def E(a):
    mol = gto.M(atom=[(s, p) for s, p in zip(a.get_chemical_symbols(), a.positions)], basis="def2-tzvpd", ecp="def2-tzvpd", charge=0, spin=0, verbose=0)
    mf = rks.RKS(mol, xc="wb97x-d3bj").density_fit(); mf.grids.level = 3; mf.conv_tol = 1e-9
    e = mf.kernel(); return float(e) * 27.211386245988, bool(mf.converged)
out = json.load(open("dft_pm7geo.json")) if os.path.exists("dft_pm7geo.json") else {}
for f in sorted(glob.glob("pm7opt/*.xyz"), key=lambda f: len(ase.io.read(f))):
    k = os.path.basename(f)[:-4]
    if k in out: continue
    e, c = E(ase.io.read(f)); out[k] = {"E_eV": e, "conv": c}; json.dump(out, open("dft_pm7geo.json", "w"), indent=1)
    print(k, c, flush=True)
# TMA·H2O: GFN2 付加体 → PM7 最終構造の直線補間
a = ase.io.read("geo/TMA_H2O.xyz"); b = ase.io.read("pm7opt/TMA_H2O.xyz")
from ase.build import minimize_rotation_and_translation
minimize_rotation_and_translation(a, b)
fr = []
for t in np.linspace(0, 1, 9):
    c = a.copy(); c.positions = (1 - t) * a.positions + t * b.positions; fr.append(c)
ase.io.write("tma_path.xyz", fr)
path = [E(c) for c in fr]
json.dump({"t": np.linspace(0, 1, 9).tolist(), "E_eV": [p[0] for p in path], "conv": [p[1] for p in path]}, open("tma_path_dft.json", "w"), indent=1)
print("path done", flush=True)
