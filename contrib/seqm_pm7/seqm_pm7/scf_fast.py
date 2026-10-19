"""SCF の高速化：適応混合 → Pulay DIIS を、同じ積分（problem）のまま 1 回の SCF の中で切り替える

これまで（v0.2）は適応混合の前処理を別の Energy 計算として流していたため、
積分と Hcore を 2 回作っていた。ここでは PYSEQM の _solve_scf を差し替え、
hybrid(n) が有効なあいだ scf_converger=[2] の呼び出しを
「scf_forward1 を最大 n 回 → その密度から scf_forward2」に置き換える。
"""
import contextlib, contextvars
import seqm.seqm_functions.scf_loop as _SL

_PRE = contextvars.ContextVar("seqm_pm7_hybrid_pre", default=0)
_ORIG = _SL._solve_scf
STATS = {"fock": 0}
_ORIG_BUILD_FOCK = _SL.build_fock


def _counting_build_fock(problem, configuration, density):
    STATS["fock"] += 1
    return _ORIG_BUILD_FOCK(problem, configuration, density)


def _solve_scf(problem, configuration, P, *, backward=False, verbose=True):
    from . import scf_compact as SC
    n = _PRE.get()
    conv = configuration.converger[0]
    ok = SC.usable(problem, configuration, backward) and conv in (1, 2)
    if conv == 1 and ok:
        r = SC.adaptive(problem, P, configuration, _SL.MAX_ITER)
        if r is not None:
            return r
    if conv != 2 or problem.unrestricted:
        return _ORIG(problem, configuration, P, backward=backward, verbose=verbose)
    if n > 0:
        r = SC.adaptive(problem, P.clone(), configuration, n) if ok else None
        if r is None:
            old = _SL.MAX_ITER
            _SL.MAX_ITER = n
            try:
                r = _SL.scf_forward1(problem, P.clone(), configuration, backward=backward, verbose=False)
            finally:
                _SL.MAX_ITER = old
        P1, nc1 = r
        if not bool(nc1.any()):
            return P1, nc1                      # 適応混合だけで収束した
        P = P1
    if ok:
        r = SC.diis(problem, P, configuration, _SL.MAX_ITER)
        if r is not None:
            return r
    return _SL.scf_forward2(problem, P, configuration, backward=backward, verbose=verbose)


@contextlib.contextmanager
def hybrid(n):
    tok = _PRE.set(int(n))
    try:
        yield
    finally:
        _PRE.reset(tok)


def patch():
    _SL._solve_scf = _solve_scf
    _SL.build_fock = _counting_build_fock
