import os, glob, json, time, ase.io, warnings
warnings.filterwarnings("ignore")
from pyscf import gto
from gpu4pyscf.dft import rks
out = json.load(open("dft.json")) if os.path.exists("dft.json") else {}
files = sorted(glob.glob("geo/*.xyz"), key=lambda f: len(ase.io.read(f)))
for f in files:
    k = os.path.basename(f)[:-4]
    if k in out: continue
    a = ase.io.read(f)
    mol = gto.M(atom=[(s, p) for s, p in zip(a.get_chemical_symbols(), a.positions)], basis="def2-tzvpd", ecp="def2-tzvpd",
                charge=0, spin=0, verbose=0)
    mf = rks.RKS(mol, xc="wb97x-d3bj").density_fit(); mf.grids.level = 3; mf.conv_tol = 1e-9
    t = time.time(); E = mf.kernel()
    out[k] = {"E_eV": float(E) * 27.211386245988, "conv": bool(mf.converged), "nao": int(mol.nao), "t": time.time() - t}
    json.dump(out, open("dft.json", "w"), indent=1)
    print(k, len(a), "nao", mol.nao, "conv", mf.converged, f"{time.time()-t:.0f}s", flush=True)
