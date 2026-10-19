"""validation/mopac_ref_d.json の全系（22 分子 × PM6 / PM7- / PM7）を計算し、per_system_d.csv と summary_d.txt を書く"""
import json, os, sys, time, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import seqm_pm7, torch
torch.set_num_threads(int(os.environ.get("NTHREADS", "4")))
V = os.path.join(os.path.dirname(__file__), "..", "validation")
REF = json.load(open(os.path.join(V, "mopac_ref_d.json")))
rows = []
for nm, r in REF.items():
    for m in ("PM6", "PM7-", "PM7"):
        t0 = time.time(); q = seqm_pm7.compute(r["sym"], np.array(r["x"]), m, charge=r["charge"]); dt = time.time() - t0
        dH = q["Hf"] - r[m]["Hf"]; dF = np.abs(-q["forces"] - np.array(r[m]["grad"])).max()
        rows.append((nm, len(r["sym"]), m, r[m]["Hf"], q["Hf"], dH, dF, q["scf_converged"], "/".join(map(str, q["scf_converger"])), dt))
        print(f"{nm:12s} {m:5s} dH {dH:+.4f} dF {dF:.4f} {dt:5.1f} s", flush=True)
with open(os.path.join(V, "per_system_d.csv"), "w") as f:
    f.write("system,natoms,method,Hf_mopac,Hf_seqm_pm7,dHf,max_abs_dF,scf_converged,scf_converger,time_s\n")
    for t in rows: f.write(",".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in t) + "\n")
with open(os.path.join(V, "summary_d.txt"), "w") as f:
    f.write(f"seqm_pm7 v{seqm_pm7.__version__} と MOPAC 23.2.5 の照合（d 元素を含む 22 分子、1SCF、xTB 最適化構造）\n")
    for m in ("PM6", "PM7-", "PM7"):
        sel = [t for t in rows if t[2] == m]
        nN = [t for t in sel if not ("N" in REF[t[0]]["sym"] and m == "PM6")]
        f.write(f"{m:5s}: max|dHf| = {max(abs(t[5]) for t in sel):.4f} kcal/mol, max|dF| = {max(t[6] for t in sel):.4f} kcal/mol/Å"
                f"（N を含む PM6 を除くと {max(t[6] for t in nN):.4f}）, SCF 収束 {sum(t[7] for t in sel)}/{len(sel)}\n")
print(open(os.path.join(V, "summary_d.txt")).read())
