"""d 軌道をもつ元素（Al, Si, Zr, Hf など）の MOPAC 23.2.5 との照合（validation/mopac_ref_d.json）。pytest で実行"""
import json, os, numpy as np, pytest
import seqm_pm7
REF = json.load(open(os.path.join(os.path.dirname(__file__), "..", "validation", "mopac_ref_d.json")))
PICK = ["SiH4", "AlMe3", "SiOH4", "ZrOH4", "HfMe4", "Zr2O2OH4", "HfCl4"]

@pytest.mark.parametrize("name", PICK)
@pytest.mark.parametrize("method", ["PM6", "PM7-", "PM7"])
def test_energy_d(name, method):
    r = REF[name]; q = seqm_pm7.compute(r["sym"], np.array(r["x"]), method, charge=r["charge"])
    assert q["scf_converged"]
    assert abs(q["Hf"] - r[method]["Hf"]) < 5e-3

@pytest.mark.parametrize("name,method", [("ZrOH4", "PM7"), ("SiOH4", "PM7"), ("AlMe3", "PM6")])
def test_forces_d(name, method):
    """力は MOPAC の勾配と 0.05 kcal/mol/Å 以内（PM6 の N を含む系は MOPAC の nsp2 勾配が近似なので除く）"""
    r = REF[name]; q = seqm_pm7.compute(r["sym"], np.array(r["x"]), method)
    assert np.abs(-q["forces"] - np.array(r[method]["grad"])).max() < 0.05

def test_rotation_invariance_d_d_pairs():
    """d 軌道どうしの組（Si–Cl, Cl–Cl）を含む系のエネルギーが向きに依らない（PYSEQM 本体では最大 0.24 kcal/mol 変わる）"""
    from scipy.spatial.transform import Rotation as R
    T = np.array([(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]) / np.sqrt(3)
    x = np.vstack([[0, 0, 0], 2.03 * T]) + np.random.default_rng(7).normal(0, 0.15, (5, 3))
    E = [seqm_pm7.compute(["Si"] + ["Cl"] * 4, R.random(random_state=k).apply(x), "PM6")["Hf"] for k in range(3)]
    assert max(E) - min(E) < 1e-4      # SCF の収束判定程度（PYSEQM 本体のままだと 0.24）

def test_pm7_si_supported():
    q = seqm_pm7.compute(["Si", "H", "H", "H", "H"], np.array([[0, 0, 0], [0.86, 0.86, 0.86], [-0.86, -0.86, 0.86], [-0.86, 0.86, -0.86], [0.86, -0.86, -0.86]]), "PM7")
    assert np.isfinite(q["Hf"])
