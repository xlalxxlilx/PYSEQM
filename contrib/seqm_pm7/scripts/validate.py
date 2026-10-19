"""seqm_pm7 と MOPAC 23 の照合（エネルギー・補正項の内訳・勾配）"""
import os, re, sys, json, subprocess, tempfile, numpy as np, ase.io
from concurrent.futures import ProcessPoolExecutor
from scipy.spatial.transform import Rotation as R
from ase.build import molecule
from ase.collections import g2, s22
MOPAC = os.path.expanduser(os.environ.get("MOPAC_HOME", "~/opt/mopac-23.2.5-linux") + "/bin/mopac")

def mopac(job):
    s, x, kw, charge = job
    d = tempfile.mkdtemp(prefix="mopv_")
    with open(f"{d}/m.mop", "w") as f:
        f.write(f"{kw} 1SCF CHARGE={charge} GRAD PRECISE DISP AUX(PRECISION=9)\n\n\n")
        for a, c in zip(s, x): f.write(f"{a} {c[0]:.8f} 1 {c[1]:.8f} 1 {c[2]:.8f} 1\n")
    subprocess.run([MOPAC, "m.mop"], cwd=d, capture_output=True, env={**os.environ, "OMP_NUM_THREADS": "1", "LD_LIBRARY_PATH": os.path.expanduser(os.environ.get("MOPAC_HOME", "~/opt/mopac-23.2.5-linux") + "/lib")})
    out = open(f"{d}/m.out").read(); aux = open(f"{d}/m.aux").read()
    Hf = float(re.search(r"HEAT_OF_FORMATION:KCAL/MOL=([-+0-9.EeD]+)", aux).group(1).replace("D", "E"))
    G = np.array(re.search(r"GRADIENTS:KCAL/MOL/ANGSTROM\[\d+\]=\s*(.*?)\n\s*[A-Z]", aux, re.S).group(1).split(), float).reshape(-1, 3)
    ed = re.search(r"DISPERSION ENERGY\s*=\s*([-0-9.]+)", out); eh = re.search(r"H-BOND ENERGY\s*=\s*([-0-9.]+)", out)
    return {"Hf": Hf, "G": G.tolist(), "E_disp": float(ed.group(1)) if ed else 0.0, "E_hb": float(eh.group(1)) if eh else 0.0}

def testset():
    rng = np.random.default_rng(7); T = []
    ok = set("HCNOF")
    for nm in g2.names:
        a = g2[nm]
        if set(a.get_chemical_symbols()) <= ok and len(a) > 1 and sum(a.get_initial_magnetic_moments()) == 0:
            T.append(("g2:" + nm, a.get_chemical_symbols(), a.positions, 0))
    for nm in s22.names:
        a = s22[nm]; T.append(("s22:" + nm, a.get_chemical_symbols(), a.positions, 0))
    W = "../pyseqm_test/clusters/"
    for t in ["HF_w4", "HF_w8", "NH3_w4", "NH3_w8", "HF_w20_start_neutral", "HF_w20_start_ionpair", "NH3_w20_start_neutral", "NH3_w20_start_ionpair"]:
        f = W + (t + "_gfn2.xyz" if "start" not in t else t + ".xyz")
        a = ase.io.read(f); T.append(("cl:" + t, a.get_chemical_symbols(), a.positions, 0))
    # イオン
    T.append(("ion:H3O+", ["O","H","H","H"], np.array([[0,0,0.05],[0,0.97,-0.22],[0.84,-0.485,-0.22],[-0.84,-0.485,-0.22]]), 1))
    T.append(("ion:NH4+", ["N","H","H","H","H"], np.array([[0,0,0],[0.59,0.59,0.59],[-0.59,-0.59,0.59],[-0.59,0.59,-0.59],[0.59,-0.59,-0.59]]), 1))
    T.append(("ion:H5O2+", ["O","H","H","H","O","H","H"], np.array([[-1.2,0,0],[-1.6,0.8,0.3],[-1.6,-0.8,0.3],[0.0,0.05,0.0],[1.2,0,0],[1.6,0.8,-0.3],[1.6,-0.8,-0.3]]), 1))
    # ランダム回転（PYSEQM の軸揃いの不具合を避け、両者に同じ座標を渡す）＋ 小さな歪み
    out = []
    for nm, s, x, c in T:
        x = R.random(random_state=rng.integers(1e9)).apply(np.asarray(x, float) - np.mean(x, 0)) + rng.normal(0, 0.01, (len(s), 3))
        out.append((nm, list(s), x, c))
    return out

if __name__ == "__main__":
    T = testset(); print("test systems:", len(T), flush=True)
    jobs = [(s, x, kw, c) for nm, s, x, c in T for kw in ("PM6", "PM7-", "PM7")]
    with ProcessPoolExecutor(8) as ex: res = list(ex.map(mopac, jobs, chunksize=4))
    M = {}
    for (nm, s, x, c), i in zip(T, range(0, len(jobs), 3)):
        M[nm] = {"sym": s, "x": x.tolist(), "charge": c, "PM6": res[i], "PM7-": res[i+1], "PM7": res[i+2]}
    json.dump(M, open("mopac_ref.json", "w")); print("saved mopac_ref.json")
