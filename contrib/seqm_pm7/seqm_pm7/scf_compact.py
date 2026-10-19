"""実際の軌道だけの行列で SCF を回す（PYSEQM の scf_forward1 / scf_forward2 と同じアルゴリズム）

PYSEQM の d 経路は全原子を 9 軌道分（sp 経路は 4 軌道分）に詰めた行列を使い、DIIS の交換子 FP − PF、
Fock の外挿、固有値問題の前の詰め直し（packd）をその大きさで行う。148 原子では 1332 次元だが、実際の軌道は 324。
ここでは反復の中身を実際の軌道（packd と同じ並び：d をもつ原子 9、他の重原子 4、H 1）だけで行い、
Fock を作るときだけ 9 軌道分に戻す。詰めた部分は P も交換子もゼロなので、和・最大値・内積は元と同じ。
閉殻で、まとめて計算する構造の原子構成がそろっているときだけ使う（それ以外は元の関数）。
"""
import torch
import seqm.seqm_functions.scf_loop as _SL

ENABLED = True


def _index(problem, nbf):
    nsh, nh, nhy = problem.nSuperHeavy, problem.nHeavy, problem.nHydro
    if not (bool((nsh == nsh[0]).all()) and bool((nh == nh[0]).all()) and bool((nhy == nhy[0]).all())):
        return None
    if not bool((problem.nOccMO == problem.nOccMO[0]).all()):
        return None
    a, b, c = int(nsh[0]), int(nh[0]), int(nhy[0])
    idx = []
    at = 0
    if nbf == 9:
        for _ in range(a): idx += [9 * at + k for k in range(9)]; at += 1
        for _ in range(b): idx += [9 * at + k for k in range(4)]; at += 1
        for _ in range(c): idx += [9 * at]; at += 1
    else:
        if a: return None
        for _ in range(b): idx += [4 * at + k for k in range(4)]; at += 1
        for _ in range(c): idx += [4 * at]; at += 1
    return torch.tensor(idx, device=problem.M.device)


class _Ctx:
    def __init__(self, problem, configuration, P):
        self.problem, self.configuration = problem, configuration
        self.nbas = P.shape[-1]
        self.nbf = 9 if configuration.method == "PM6" else 4
        self.idx = _index(problem, self.nbf)
        if self.idx is not None:
            self.nocc = int(problem.nOccMO[0])

    def g(self, X):                      # 9 軌道分 → 実際の軌道
        return X[:, self.idx][:, :, self.idx]

    def s(self, Xc):                     # 実際の軌道 → 9 軌道分
        n = Xc.shape[0]
        out = Xc.new_zeros(n, self.nbas, self.nbas)
        out[:, self.idx.unsqueeze(1), self.idx.unsqueeze(0)] = Xc
        return out

    def fock(self, Pc):
        F = _SL.build_fock(self.problem, self.configuration, self.s(Pc))
        return self.g(F)

    def density(self, Fc):
        e, v = torch.linalg.eigh(Fc, UPLO="U")
        vo = v[:, :, :self.nocc]
        return 2.0 * vo @ vo.transpose(1, 2)


def usable(problem, configuration, backward):
    if not ENABLED or backward or problem.unrestricted or configuration.sp2[0]:
        return None
    if problem.M.dim() != 3:
        return None
    return True


def adaptive(problem, P, configuration, max_iter):
    """scf_forward1（適応混合）。戻り値 (P, notconverged)。使えないときは None"""
    ctx = _Ctx(problem, configuration, P)
    if ctx.idx is None:
        return None
    M = problem.M
    nmol, molsize, eps = problem.nmol, problem.molsize, configuration.eps
    Pc = ctx.g(P).clone()
    Hc = ctx.g(_SL.reshape_Hcore(M, nmol, molsize, configuration.method))
    Fc = ctx.fock(Pc)
    Pold = torch.zeros_like(Pc)
    err = torch.ones(nmol, dtype=P.dtype, device=P.device)
    dm_err = torch.ones_like(err); dm_element_err = torch.ones_like(err)
    notconverged = torch.ones(nmol, dtype=torch.bool, device=P.device)
    Eelec = _SL.elec_energy(Pc, Fc, Hc); Eelec_new = torch.zeros_like(Eelec)
    Pold2_diag = torch.zeros(Pc.shape[:-1], dtype=P.dtype, device=P.device)
    mss = problem.nSuperHeavy * 9 + problem.nHeavy * 4 + problem.nHydro * 4
    for k in range(1, max_iter + 1):
        nz = notconverged.nonzero(as_tuple=False).squeeze(-1)
        Pnew = ctx.density(Fc[nz])
        Pmix, diag_prev = _SL.adaptive_mix(k, Pc[nz], Pnew, Pold2_diag[nz], False)
        Pold[notconverged] = Pc[notconverged]
        Pc[notconverged] = Pmix
        Pold2_diag[notconverged] = diag_prev
        Fc = ctx.fock(Pc)
        Eelec_new[notconverged] = _SL.elec_energy(Pc[notconverged], Fc[notconverged], Hc[notconverged])
        notconverged, _, _ = _SL.measure_convergence(Pold, Pc, notconverged, mss, dm_err, dm_element_err,
                                                     Eelec_new, err, Eelec, eps)
        Eelec[notconverged] = Eelec_new[notconverged]
        if int(notconverged.sum()) == 0:
            break
    return ctx.s(Pc), notconverged


def diis(problem, P, configuration, max_iter):
    """scf_forward2（Pulay DIIS、nDirect1 = nAdapt = 0）。戻り値 (P, notconverged)。使えないときは None"""
    ctx = _Ctx(problem, configuration, P)
    if ctx.idx is None:
        return None
    M = problem.M
    nmol, molsize, eps = problem.nmol, problem.molsize, configuration.eps
    dtype, device = P.dtype, P.device
    Pc = ctx.g(P).clone()
    Fc = ctx.fock(Pc)
    Hc = ctx.g(_SL.reshape_Hcore(M, nmol, molsize, configuration.method))
    notconverged = torch.ones(nmol, dtype=torch.bool, device=device)
    err = torch.ones(nmol, dtype=dtype, device=device)
    Pold = torch.zeros_like(Pc)
    dm_err = torch.ones_like(err); dm_element_err = torch.ones_like(err)
    Eelec = _SL.elec_energy(Pc, Fc, Hc); Eelec_new = torch.zeros_like(Eelec)
    mss = problem.nSuperHeavy * 9 + problem.nHeavy * 4 + problem.nHydro * 4
    nb = Pc.shape[-1]
    nFock = 10
    tri_i, tri_j = torch.triu_indices(nb, nb, device=device)
    FPPF = torch.zeros(nmol, nFock, tri_i.numel(), dtype=dtype, device=device)
    def fresh_emat():
        return (torch.eye(nFock + 1, nFock + 1, dtype=dtype, device=device) - 1.0).expand(nmol, nFock + 1, nFock + 1).tril().clone()
    EMAT = fresh_emat()
    FOCK = torch.zeros(nmol, nFock, nb, nb, dtype=dtype, device=device)
    counter, cFock = -1, 0
    alpha_direct = 0.5
    diis_error = torch.empty_like(Eelec).fill_(torch.finfo(dtype).max)
    eval_eps, clamp_eps = (1e-13, 1e-15) if dtype == torch.float64 else (1e-5, 1e-7)
    Nnot = nmol
    for k in range(0, max_iter + 1):
        if Nnot == 0:
            return ctx.s(Pc), notconverged
        cFock = cFock + 1 if cFock < nFock else nFock
        counter = (counter + 1) % nFock
        FOCK[notconverged, counter] = Fc[notconverged]
        reset = False
        with torch.no_grad():
            F_nc = Fc[notconverged]; P_nc = Pc[notconverged]
            Cm = F_nc @ P_nc - P_nc @ F_nc
            Cp = Cm[:, tri_i, tri_j]
            FPPF[notconverged, counter] = Cp
            diis_error[notconverged] = Cp.abs().amax(dim=1)
            prod = torch.bmm(FPPF[notconverged, counter:counter + 1], FPPF[notconverged, :cFock].transpose(1, 2))
            EMAT[notconverged, counter, :cFock] = prod[:, 0, :]
        if cFock >= 2:
            with torch.no_grad():
                EVEC = EMAT[notconverged, :cFock + 1, :cFock + 1]
                denom = EVEC[:, counter, counter].clamp(clamp_eps)
                EVEC[:, :cFock, :cFock] /= denom.view(-1, 1, 1)
                L, Q = torch.linalg.eigh(EVEC)
                absv = L.abs()
                cond = absv.amax(dim=-1) / absv.amin(dim=-1)
                reset = bool(torch.any(cond > 1e7))
                valid = absv > eval_eps
                inv = torch.zeros_like(L); inv[valid] = L[valid].reciprocal()
                coeff = -torch.einsum("bki,bi,bi->bk", Q[:, :cFock, :], inv, Q[:, -1, :])
            Fc[notconverged] = torch.sum(FOCK[notconverged, :cFock] * coeff.unsqueeze(-1).unsqueeze(-1), dim=1)
        Pnew = ctx.density(Fc[notconverged])
        Pold[notconverged] = Pc[notconverged]
        if cFock < 2:
            Pc[notconverged] = alpha_direct * Pc[notconverged] + (1.0 - alpha_direct) * Pnew
        else:
            Pc[notconverged] = Pnew
        Fc = ctx.fock(Pc)
        Eelec_new[notconverged] = _SL.elec_energy(Pc[notconverged], Fc[notconverged], Hc[notconverged])
        notconverged, _, _ = _SL.measure_convergence(Pold, Pc, notconverged, mss, dm_err, dm_element_err,
                                                     Eelec_new, err, Eelec, eps, diis_error=diis_error)
        Eelec[notconverged] = Eelec_new[notconverged]
        Nnot = int(notconverged.sum())
        if reset:
            counter, cFock = -1, 0
            FPPF.zero_(); EMAT = fresh_emat(); FOCK.zero_()
    return ctx.s(Pc), notconverged
