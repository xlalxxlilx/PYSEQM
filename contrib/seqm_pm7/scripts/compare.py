import json, sys, time, numpy as np, torch
torch.set_num_threads(4)
import seqm_pm7
M = json.load(open("mopac_ref.json"))
names = list(M) if len(sys.argv) < 2 else [n for n in M if any(n.startswith(p) for p in sys.argv[1].split(","))]
out = {}
for nm in names:
    r = M[nm]; s, x, c = r["sym"], np.array(r["x"]), r["charge"]; out[nm] = {}
    for meth in ["PM6", "PM7-", "PM7"]:
        t = time.time(); q = seqm_pm7.compute(s, x, meth, charge=c)
        ref = r[meth]; G = np.array(ref["G"])
        out[nm][meth] = {"dHf": float(q["Hf"] - ref["Hf"]), "dF": float(np.abs(-q["forces"] - G).max()),
                         "E_disp": float(q["E_disp"]), "E_hb": float(q["E_hb"]), "ref_disp": ref["E_disp"], "ref_hb": ref["E_hb"],
                         "E_CC": float(q["E_CC"]), "E_NHCO": float(q["E_NHCO"]), "E_NSP2": float(q["E_NSP2"]), "t": time.time() - t}
    o = out[nm]
    print(f"{nm:34s} PM6 {o['PM6']['dHf']:+8.3f} | PM7- {o['PM7-']['dHf']:+8.3f} | PM7 {o['PM7']['dHf']:+8.3f}  "
          f"disp {o['PM7']['E_disp']-o['PM7']['ref_disp']:+7.3f} hb {o['PM7']['E_hb']-o['PM7']['ref_hb']:+7.3f}  "
          f"maxdF {o['PM6']['dF']:.2f}/{o['PM7-']['dF']:.2f}/{o['PM7']['dF']:.2f}", flush=True)
json.dump(out, open("compare_out.json" if len(sys.argv) < 2 else f"compare_{sys.argv[1].replace(',','_').replace(':','')}.json", "w"), indent=1)
