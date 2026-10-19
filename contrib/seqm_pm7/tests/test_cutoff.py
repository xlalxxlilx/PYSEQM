"""距離による打ち切り（cutoff）が打ち切りなしの結果と一致すること。pytest で実行"""
import json, os, numpy as np, pytest
import seqm_pm7

V = os.path.join(os.path.dirname(__file__), "..", "validation")
REF = json.load(open(os.path.join(V, "mopac_ref.json")))
REFD = json.load(open(os.path.join(V, "mopac_ref_d.json")))


def _pair(r, shift):
    s = r["sym"]; x = np.array(r["x"]); x = x - x.mean(0)
    return s + s, np.vstack([x, x + np.array([shift, 0.0, 0.0])])


@pytest.mark.parametrize("name,ref,shift", [("cl:HF_w8", "sp", 13.0), ("TDMAZ", "d", 15.0), ("Zr4O6OH4", "d", 12.0)])
def test_cutoff_matches_full(name, ref, shift):
    r = (REF if ref == "sp" else REFD)[name]
    s, x = _pair(r, shift)
    q0 = seqm_pm7.compute(s, x, "PM7")
    q1 = seqm_pm7.compute(s, x, "PM7", cutoff=10.0)
    assert abs(q1["Hf"] - q0["Hf"]) < 1e-3
    assert np.abs(q1["forces"] - q0["forces"]).max() < 1e-3


def test_cutoff_guards():
    r = REF["s22:Water_dimer"]; x = np.array(r["x"])
    with pytest.raises(ValueError):
        seqm_pm7.compute(r["sym"], x, "PM7", cutoff=6.0)
    with pytest.raises(ValueError):
        seqm_pm7.compute(r["sym"], x, "PM6", cutoff=10.0)
