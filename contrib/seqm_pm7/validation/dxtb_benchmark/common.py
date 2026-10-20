import os, torch, dxtb, numpy as np, ase.io, time
AA2AU = 1.0 / 0.52917721090380; EH2EV = 27.211386245988
def calc(numbers, gfn=2, device="cpu", dtype=torch.float64, **opt):
    par = dxtb.GFN2_XTB if gfn == 2 else dxtb.GFN1_XTB
    o = {"verbosity": 0, "maxiter": 300, "x_atol": 1e-7, "f_atol": 1e-7}
    if os.environ.get("DXTB_GUESS"): o["guess"] = os.environ["DXTB_GUESS"]; o.update(opt)
    return dxtb.Calculator(numbers, par, opts=o, device=torch.device(device), dtype=dtype)
def ef(atoms_list, gfn=2, device="cpu", dtype=torch.float64, chrg=0, **opt):
    """atoms_list: 同じ原子数・同じ順の構造（まとめ計算は dxtb の batch_mode）。戻り値 E [eV], F [eV/Å]"""
    dev = torch.device(device)
    Z = torch.tensor(np.array([a.numbers for a in atoms_list]), device=dev)
    X = torch.tensor(np.array([a.positions for a in atoms_list]) * AA2AU, device=dev, dtype=dtype, requires_grad=True)
    if len(atoms_list) == 1: Z, Xin = Z[0], X[0]
    else: Xin = X
    c = calc(Z, gfn, device, dtype, **opt)
    q = chrg if len(atoms_list) == 1 else torch.full((len(atoms_list),), float(chrg), device=dev, dtype=dtype)
    E = c.get_energy(Xin, chrg=q)
    g, = torch.autograd.grad(E.sum(), X)
    return (E.detach().cpu().numpy() * EH2EV).reshape(-1), -(g.detach().cpu().numpy() * EH2EV * AA2AU)
