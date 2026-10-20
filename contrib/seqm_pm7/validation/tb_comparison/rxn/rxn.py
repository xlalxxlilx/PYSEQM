import json, numpy as np
D = json.load(open("dft.json")); X = json.load(open("xtb.json")); P = json.load(open("pm7.json"))
EV = 96.485332; KC = 4.184
R = {
 "R1 ZrMe4+H2O→ZrMe3OH+CH4":         ({"ZrMe4": 1, "H2O": 1}, {"ZrMe3OH": 1, "CH4": 1}),
 "R2 HfMe4+H2O→HfMe3OH+CH4":         ({"HfMe4": 1, "H2O": 1}, {"HfMe3OH": 1, "CH4": 1}),
 "R3 AlMe3+H2O→AlMe2OH+CH4":         ({"AlMe3": 1, "H2O": 1}, {"AlMe2OH": 1, "CH4": 1}),
 "R4 TDMAZ+H2O→Zr(NMe2)3OH+HNMe2":   ({"TDMAZ": 1, "H2O": 1}, {"ZrNMe23OH": 1, "HNMe2": 1}),
 "R5 TDMAH+H2O→Hf(NMe2)3OH+HNMe2":   ({"TDMAH": 1, "H2O": 1}, {"HfNMe23OH": 1, "HNMe2": 1}),
 "R6 CpZr(NMe2)3+H2O→CpZr(NMe2)2OH+HNMe2": ({"CpZrNMe23": 1, "H2O": 1}, {"CpZrNMe22OH": 1, "HNMe2": 1}),
 "R7 HfCl4+H2O→HfCl3OH+HCl":         ({"HfCl4": 1, "H2O": 1}, {"HfCl3OH": 1, "HCl": 1}),
 "R8 SiH2(NMe2)2+H2O→SiH2(NMe2)OH+HNMe2": ({"SiH2NMe22": 1, "H2O": 1}, {"SiH2NMe2OH": 1, "HNMe2": 1}),
 "R9 ZrMe4+4H2O→Zr(OH)4+4CH4":       ({"ZrMe4": 1, "H2O": 4}, {"ZrOH4": 1, "CH4": 4}),
 "R10 HfMe4+4H2O→Hf(OH)4+4CH4":      ({"HfMe4": 1, "H2O": 4}, {"HfOH4": 1, "CH4": 4}),
 "R11 TDMAZ+4H2O→Zr(OH)4+4HNMe2":    ({"TDMAZ": 1, "H2O": 4}, {"ZrOH4": 1, "HNMe2": 4}),
 "R12 2Zr(OH)4→Zr2O2(OH)4+2H2O":     ({"ZrOH4": 2}, {"Zr2O2OH4": 1, "H2O": 2}),
 "R13 AlMe3+H2O→AlMe3·H2O":          ({"AlMe3": 1, "H2O": 1}, {"TMA_H2O": 1}),
}
def e(meth, k):
    if meth == "DFT": return D[k]["E_eV"] * EV
    if meth in ("GFN1", "GFN2"): return X[k][meth] * EV
    return P[k][meth] * KC
def rxn(meth, r):
    a, b = r; return sum(n * e(meth, k) for k, n in b.items()) - sum(n * e(meth, k) for k, n in a.items())
meths = ["DFT", "GFN2", "GFN1", "PM7", "PM6", "PM7_own"]
tab = {nm: {m: rxn(m, r) for m in meths} for nm, r in R.items()}
print(f"{'反応':42s}" + "".join(f"{m:>9s}" for m in meths))
for nm, v in tab.items(): print(f"{nm:42s}" + "".join(f"{v[m]:9.1f}" for m in meths))
stats = {}
for m in meths[1:]:
    keys = [k for k in tab if not (m == "PM7_own" and k.startswith("R13"))]
    err = np.array([tab[k][m] - tab[k]["DFT"] for k in keys])
    stats[m] = {"MAE": float(np.abs(err).mean()), "MSE": float(err.mean()), "max": float(np.abs(err).max()), "n": len(keys)}
    print(f"{m:8s} MAE {stats[m]['MAE']:6.1f}  平均符号付き誤差 {stats[m]['MSE']:+6.1f}  最大 {stats[m]['max']:6.1f} kJ/mol（{len(keys)} 反応）")
json.dump({"reactions": tab, "stats": stats}, open("rxn_results.json", "w"), indent=1, ensure_ascii=False)
