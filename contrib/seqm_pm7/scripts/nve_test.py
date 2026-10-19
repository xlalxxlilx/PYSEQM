import numpy as np, ase.io, json, sys
from ase import units
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from scipy.spatial.transform import Rotation as R
from seqm_pm7 import SeqmPM7
meth = sys.argv[1]
a = ase.io.read("../pyseqm_test/clusters/HF_w4_gfn2.xyz"); a.positions = R.from_euler("zyx", [17, 29, 41], degrees=True).apply(a.positions)
a.calc = SeqmPM7(meth)
MaxwellBoltzmannDistribution(a, temperature_K=300, rng=np.random.default_rng(1)); Stationary(a); ZeroRotation(a)
md = VelocityVerlet(a, 0.25 * units.fs); E = []; nhb = []
def rec():
    E.append(a.get_potential_energy() + a.get_kinetic_energy()); nhb.append(len(a.calc.last["topology"].nbonds) if False else a.calc.last.get("E_hb", 0.0))
md.attach(rec, interval=1); md.run(int(sys.argv[2]) if len(sys.argv)>2 else 300)
E = np.array(E) * 23.0605
dE = np.diff(E)
print(f"{meth}: total-energy drift {E[-1]-E[0]:+.3f} kcal/mol over {len(E)*0.25:.0f} fs, max step jump {np.abs(dE).max():.3f} kcal/mol, std {E.std():.3f}")
json.dump({"E": E.tolist(), "Ehb": [float(v) for v in nhb]}, open(f"nve_{meth}.json", "w"))
