"""dxtb の GFN2 パラメータについての微分：R2（HfMe4 + H2O → HfMe3OH + CH4）の反応エネルギーを Hf のパラメータで微分し、差分近似と比べる"""
import os, sys, json, time, numpy as np, ase.io, torch, dxtb
sys.path.insert(0, ".")
from common import AA2AU, EH2EV
dev = torch.device(sys.argv[1] if len(sys.argv) > 1 else "cpu")
torch.set_num_threads(4)
G = {k: ase.io.read(f"../rxn/geo/{k}.xyz") for k in ("HfMe4", "H2O", "HfMe3OH", "CH4")}
St = {"HfMe4": -1, "H2O": -1, "HfMe3OH": 1, "CH4": 1}
O = {"verbosity": 0, "maxiter": 300, "x_atol": 1e-9, "f_atol": 1e-9, "guess": "sad"}
pm = dxtb.ParamModule(dxtb.GFN2_XTB, dtype=torch.float64, device=dev)
pm.set_differentiable("element", "Hf")
P = {n.split("element.Hf.")[1].replace(".param", ""): p for n, p in pm.named_parameters() if ".Hf." in n and p.requires_grad}
def dE(with_grad=True):
    tot = 0.0
    for k, a in G.items():
        c = dxtb.Calculator(torch.tensor(a.numbers, device=dev), pm, opts=O, device=dev, dtype=torch.float64)
        tot = tot + St[k] * c.get_energy(torch.tensor(a.positions * AA2AU, device=dev, dtype=torch.float64))
    return tot * EH2EV * 96.485332      # kJ/mol
t = time.time(); E = dE(); gr = torch.autograd.grad(E, list(P.values()), allow_unused=True); t_ad = time.time() - t
res = {"dE_kJ": float(E), "t_energy_plus_paramgrad_s": t_ad, "params": {}}
h = float(os.environ.get("FD_H", "1e-4"))
with torch.no_grad():
    for (nm, p), g in zip(P.items(), gr):
        g = np.zeros(p.shape) if g is None else g.detach().cpu().numpy()
        fd = np.zeros(p.numel())
        for i in range(p.numel()):
            flat = p.view(-1); v0 = float(flat[i])
            flat[i] = v0 + h; ep = float(dE()); flat[i] = v0 - h; em = float(dE()); flat[i] = v0
            fd[i] = (ep - em) / (2 * h)
        res["params"][nm] = {"value": p.detach().cpu().numpy().reshape(-1).tolist(), "autograd": g.reshape(-1).tolist(), "fd": fd.tolist(),
                             "max_abs_diff": float(np.abs(g.reshape(-1) - fd).max())}
        print(f"{nm:8s} AD {np.round(g.reshape(-1), 4)}  FD {np.round(fd, 4)}", flush=True)
print("dE", float(E), "kJ/mol; エネルギー＋パラメータ微分", round(t_ad, 2), "s")
json.dump(res, open(f"dxtb_paramgrad_{dev.type}.json", "w"), indent=1)
