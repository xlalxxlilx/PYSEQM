import json, glob, numpy as np
R = {}
for f in sorted(glob.glob("compare_*.json")): R.update(json.load(open(f)))
print("systems:", len(R))
for meth in ["PM6", "PM7-", "PM7"]:
    dH = np.array([R[k][meth]["dHf"] for k in R]); dF = np.array([R[k][meth]["dF"] for k in R])
    print(f"{meth:5s} |ΔHf| mean {np.abs(dH).mean():.4f}  max {np.abs(dH).max():.4f} ({max(R, key=lambda k: abs(R[k][meth]['dHf']))})  "
          f"max|ΔF| median {np.median(dF):.3f}  95% {np.percentile(dF,95):.3f}  max {dF.max():.3f} ({max(R, key=lambda k: R[k][meth]['dF'])}) kcal/mol/Å")
dd = np.array([R[k]["PM7"]["E_disp"] - R[k]["PM7"]["ref_disp"] for k in R]); dh = np.array([R[k]["PM7"]["E_hb"] - R[k]["PM7"]["ref_hb"] for k in R])
hbmax = max(abs(R[k]["PM7"]["ref_hb"]) for k in R)
print(f"E_disp |Δ| max {np.abs(dd).max():.5f}   E_hb |Δ| max {np.abs(dh).max():.5f} (largest |E_hb| {hbmax:.2f} kcal/mol)  #systems with E_hb≠0: {sum(abs(R[k]['PM7']['ref_hb'])>1e-3 for k in R)}")
big = [(k, R[k]["PM7"]["dF"]) for k in R if R[k]["PM7"]["dF"] > 0.5]
print("PM7 force outliers >0.5:", big)
json.dump(R, open("validation_all.json", "w"), indent=1)
