"""PM6-DH 型の分散補正（MOPAC PM6_DH_Disp の PM7 分岐）。エネルギーは kcal/mol"""
import numpy as np, torch
from .geom import _T
PM7_DISP = dict(alpha=15.450118, s=1.226593, cscale=2.286419)

def dispersion(Z, x, nbonds, alpha=PM7_DISP["alpha"], s=PM7_DISP["s"], cscale=PM7_DISP["cscale"]):
    """nbonds は set_up_dentate（入力時の構造）。炭素の C6 は結合数 4 なら 0.95、それ以外 1.65"""
    C, R, N = (np.array(_T[k]) for k in ("disp_C", "disp_R", "disp_N"))
    ok = [i for i, z in enumerate(Z) if z <= 86 and R[z-1] != 0 and C[z-1] != 0 and N[z-1] != 0]
    if len(ok) < 2: return x.sum() * 0.0
    c6a = np.array([(0.95 if nbonds[i] == 4 else 1.65) if Z[i] == 6 else C[Z[i]-1] for i in ok])
    Na = np.array([N[Z[i]-1] for i in ok]); Ra = np.array([R[Z[i]-1] for i in ok])
    I, J = np.triu_indices(len(ok), 1)
    C6 = 2.0 * (c6a[I]**2 * c6a[J]**2 * Na[I] * Na[J]) ** (1/3) / ((c6a[I] * Na[J]**2) ** (1/3) + (c6a[J] * Na[I]**2) ** (1/3))
    R0 = (Ra[I]**3 + Ra[J]**3) / (Ra[I]**2 + Ra[J]**2) / 1000.0 * 2.0          # pm → nm
    idx = torch.tensor(ok); a = idx[torch.tensor(I)]; b = idx[torch.tensor(J)]
    r = torch.linalg.vector_norm(x[a] - x[b], dim=-1)                          # Å
    C6 = torch.as_tensor(C6, dtype=x.dtype); R0 = torch.as_tensor(R0, dtype=x.dtype)
    within = r < 100.0
    rn = r * 0.1
    damp = 1.0 / (1.0 + torch.exp(-alpha * (rn / (s * R0) - 1.0)))
    e = torch.where(within, C6 / rn**6 * damp / (1000.0 * 4.184), torch.zeros_like(r))
    return -cscale * e.sum()
