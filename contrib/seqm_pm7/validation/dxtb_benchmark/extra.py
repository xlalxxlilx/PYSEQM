import sys, json, time, torch, numpy as np
sys.argv = ["speed.py", "none"]; exec(open("speed.py").read().split("conds = sys.argv")[0])
from common import calc, AA2AU, EH2EV
out = []
def run(nm, mode, dtype, tol):
    s, x = sets[nm]; a = Atoms(s, positions=x); dev = "cuda:0"
    rec = dict(system=nm, mode=mode, dtype=str(dtype).split(".")[-1], tol=tol)
    try:
        for rep in (0, 1):
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t = time.time()
            Z = torch.tensor(a.numbers, device=dev); X = torch.tensor(a.positions * AA2AU, device=dev, dtype=dtype, requires_grad=(mode == "E+F"))
            c = calc(Z, 2, dev, dtype, guess="sad", x_atol=tol, f_atol=tol)
            E = c.get_energy(X)
            if mode == "E+F": torch.autograd.grad(E, X)
            torch.cuda.synchronize(); rec[f"t{rep}"] = time.time() - t
        rec["E_eV"] = float(E) * EH2EV; rec["peak_GB"] = torch.cuda.max_memory_allocated() / 1e9
        rec["iter"] = int(c.get_iterations(X.detach())) if mode == "E" else None
    except Exception as e: rec["error"] = type(e).__name__
    torch.cuda.empty_cache(); out.append(rec); print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.items()}, flush=True)
for nm in ("TDMAZ (37)", "4x Zr4O6(OH)4 (72)", "4x TDMAZ (148)"):
    run(nm, "E", torch.float64, 1e-7); run(nm, "E", torch.float32, 1e-5); run(nm, "E+F", torch.float32, 1e-5)
for nm in ("27x Zr4O6(OH)4 (486)", "27x TDMAZ (999)"):
    run(nm, "E", torch.float64, 1e-7)
json.dump(out, open("dxtb_extra.json", "w"), indent=1)
