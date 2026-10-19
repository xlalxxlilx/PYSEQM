import json, numpy as np, seqm_pm7
from validate import mopac
M = json.load(open("mopac_ref.json"))
for nm, meth in [("s22:Uracil_dimer_h-bonded", "PM7"), ("cl:NH3_w20_start_neutral", "PM7"), ("s22:Uracil_dimer_stack", "PM6")]:
    s, x = M[nm]["sym"], np.array(M[nm]["x"]); G = np.array(M[nm][meth]["G"])
    q = seqm_pm7.compute(s, x, meth); d = np.abs(-q["forces"] - G); at = np.unravel_index(d.argmax(), d.shape)
    h = 2e-4; xp = x.copy(); xp[at] += h; xm = x.copy(); xm[at] -= h
    fdm = (mopac((s, xp, meth, 0))["Hf"] - mopac((s, xm, meth, 0))["Hf"]) / (2*h)
    fdo = (seqm_pm7.compute(s, xp, meth)["Hf"] - seqm_pm7.compute(s, xm, meth)["Hf"]) / (2*h)
    print(f"{nm} {meth} atom {at[0]}({s[at[0]]}) dir {at[1]}: MOPAC grad {G[at]:+.3f}  MOPAC central-FD {fdm:+.3f} | ours grad {-q['forces'][at]:+.3f}  ours central-FD {fdo:+.3f}")
