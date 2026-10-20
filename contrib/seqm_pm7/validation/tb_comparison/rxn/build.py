import os, subprocess, numpy as np, ase.io
from ase import Atoms
from ase.build import molecule
from ase.neighborlist import natural_cutoffs, NeighborList
import concurrent.futures as cf
P = os.path.abspath("../../pm7d")
src = {k: f"{P}/mols/{k}/xtbopt.xyz" for k in ("ZrMe4", "HfMe4", "AlMe3", "ZrOH4", "HfOH4", "Zr2O2OH4")}
src.update({k: f"{P}/mols2/{k}/xtbopt.xyz" for k in ("TDMAZ", "TDMAH", "CpZrNMe23", "HfCl4", "SiH2NMe22", "TMA_H2O")})
MO = {"Zr": 1.95, "Hf": 1.95, "Al": 1.72, "Si": 1.65}
def hydroxylate(a, donor_el):
    a = a.copy(); s = a.get_chemical_symbols()
    m = [i for i, x in enumerate(s) if x in MO][0]
    nl = NeighborList(natural_cutoffs(a, mult=1.1), self_interaction=False, bothways=True); nl.update(a)
    don = [j for j in nl.get_neighbors(m)[0] if s[j] == donor_el][0]
    # 配位子の原子（金属を除いた連結成分）
    frag, stack = {don}, [don]
    while stack:
        i = stack.pop()
        for j in nl.get_neighbors(i)[0]:
            if j != m and j not in frag: frag.add(j); stack.append(j)
    v = a.positions[don] - a.positions[m]; v /= np.linalg.norm(v)
    pO = a.positions[m] + MO[s[m]] * v
    perp = np.cross(v, [0.3, 0.5, 0.8]); perp /= np.linalg.norm(perp)
    pH = pO + 0.97 * (np.cos(np.radians(60)) * v + np.sin(np.radians(60)) * perp)
    keep = [i for i in range(len(a)) if i not in frag]
    b = a[keep]; b += Atoms("OH", positions=[pO, pH]); return b
os.makedirs("geo", exist_ok=True)
species = {k: ase.io.read(v) for k, v in src.items()}
for parent, el, name in (("ZrMe4", "C", "ZrMe3OH"), ("HfMe4", "C", "HfMe3OH"), ("AlMe3", "C", "AlMe2OH"),
                         ("TDMAZ", "N", "ZrNMe23OH"), ("TDMAH", "N", "HfNMe23OH"), ("CpZrNMe23", "N", "CpZrNMe22OH"),
                         ("HfCl4", "Cl", "HfCl3OH"), ("SiH2NMe22", "N", "SiH2NMe2OH")):
    species[name] = hydroxylate(species[parent], el)
species["H2O"] = molecule("H2O"); species["CH4"] = molecule("CH4"); species["HCl"] = molecule("HCl")
species["HNMe2"] = Atoms("NHCCHHHHHH", positions=[[0, 0, 0], [0, 0, 1.01], [1.38, -0.4, -0.4], [-0.69, 1.2, -0.4],
                                                    [1.9, 0.5, -0.7], [1.9, -0.9, 0.4], [1.4, -1.1, -1.2],
                                                    [-1.7, 1.0, -0.8], [-0.2, 1.8, -1.1], [-0.7, 1.8, 0.5]])
def opt(item):
    k, a = item
    d = f"opt/{k}"; os.makedirs(d, exist_ok=True); ase.io.write(f"{d}/in.xyz", a)
    r = subprocess.run(["xtb", "in.xyz", "--gfn", "2", "--opt", "tight"], cwd=d, capture_output=True, text=True,
                       env={**os.environ, "OMP_NUM_THREADS": "1", "OMP_STACKSIZE": "1G"})
    ok = os.path.exists(f"{d}/xtbopt.xyz")
    if ok:
        b = ase.io.read(f"{d}/xtbopt.xyz"); ase.io.write(f"geo/{k}.xyz", b)
    return k, ok, len(a)
with cf.ProcessPoolExecutor(8) as ex:
    for k, ok, n in ex.map(opt, species.items()): print(k, n, "ok" if ok else "FAILED", flush=True)
