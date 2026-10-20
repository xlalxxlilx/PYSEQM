"""単精度の固有値計算 + 尾崎スキーム II の補正（eig_mixed）が倍精度の固有値計算と一致すること。
GEMMul8 の窓口（環境変数 SEQM_PM7_OZ2_LIB）と GPU が無い環境では飛ばす。"""
import json, os, itertools, numpy as np, pytest, torch
import seqm_pm7
from seqm_pm7 import eig_mixed as EM

pytestmark = pytest.mark.skipif(not (torch.cuda.is_available() and EM.available()), reason="GEMMul8 の窓口か GPU が無い")


def test_density_matches_fp64():
    torch.manual_seed(0); n, nocc = 1200, 600; d = "cuda"
    Q, _ = torch.linalg.qr(torch.randn(n, n, dtype=torch.float64, device=d))
    lam = torch.cat([torch.linspace(-40, -11, nocc, dtype=torch.float64, device=d), torch.linspace(1, 15, n - nocc, dtype=torch.float64, device=d)])
    F = (Q * lam) @ Q.T; F = (F + F.T) / 2
    P, how = EM.density_one(F, nocc)
    e, V = torch.linalg.eigh(F); P0 = 2 * V[:, :nocc] @ V[:, :nocc].T
    assert how == "mixed"
    assert float((P - P0).abs().max()) < 1e-10


def test_scf_matches_fp64():
    V = os.path.join(os.path.dirname(__file__), "..", "validation")
    r = json.load(open(os.path.join(V, "mopac_ref_d.json")))["TDMAZ"]
    x = np.array(r["x"]); x = x - x.mean(0)
    offs = [np.array(v) * 11.0 for v in itertools.product(range(2), repeat=3)]
    s = r["sym"] * 8; X = np.vstack([x + o for o in offs])                # 648 軌道
    old = EM.MIN_N; EM.MIN_N = 500
    try:
        q0 = seqm_pm7.compute(s, X, "PM7", device="cuda", eig="fp64")
        q1 = seqm_pm7.compute(s, X, "PM7", device="cuda", eig="auto")
    finally:
        EM.MIN_N = old
    assert abs(q1["Hf"] - q0["Hf"]) < 1e-4
    assert np.abs(q1["forces"] - q0["forces"]).max() < 1e-5
