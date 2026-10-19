"""MOPAC 23.2.5 の参照値（validation/mopac_ref.json）との照合テスト。pytest で実行"""
import json, os, numpy as np, pytest
import seqm_pm7
REF = json.load(open(os.path.join(os.path.dirname(__file__), "..", "validation", "mopac_ref.json")))
PICK = ["g2:C2H2", "g2:H2CCO", "g2:CH3CONH2", "g2:C6H6", "s22:Water_dimer", "s22:Formic_acid_dimer",
        "s22:Adenine-thymine_Watson-Crick_complex", "cl:HF_w8", "cl:NH3_w8", "ion:H5O2+", "ion:NH4+"]
@pytest.mark.parametrize("name", PICK)
@pytest.mark.parametrize("method", ["PM6", "PM7-", "PM7"])
def test_energy(name, method):
    r = REF[name]; q = seqm_pm7.compute(r["sym"], np.array(r["x"]), method, charge=r["charge"])
    assert abs(q["Hf"] - r[method]["Hf"]) < 1e-3
    if method == "PM7":
        assert abs(q["E_hb"] - r["PM7"]["E_hb"]) < 1e-3 and abs(q["E_disp"] - r["PM7"]["E_disp"]) < 1e-3

@pytest.mark.parametrize("name", ["s22:Water_dimer", "cl:HF_w4", "s22:Formamide_dimer"])
def test_forces_are_energy_gradient(name):
    """力は自前のエネルギーの中心差分と一致する（MOPAC の補正項の勾配は前進差分の近似なので比較しない）"""
    r = REF[name]; s, x = r["sym"], np.array(r["x"]); q = seqm_pm7.compute(s, x, "PM7")
    rng = np.random.default_rng(0)
    for _ in range(3):
        a, k = rng.integers(len(s)), rng.integers(3); h = 1e-4
        xp = x.copy(); xp[a, k] += h; xm = x.copy(); xm[a, k] -= h
        fd = (seqm_pm7.compute(s, xp, "PM7")["Hf"] - seqm_pm7.compute(s, xm, "PM7")["Hf"]) / (2 * h)
        assert abs(-q["forces"][a, k] - fd) < 0.01
