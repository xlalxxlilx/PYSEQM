"""PYSEQM の原子定数表の修正と、d 電子の孤立原子エネルギー（EISOL の d 項）

1. PYSEQM の Constants はランタノイドを飛ばした並びで、添字 57 以降が実際の原子番号と
   ずれている（添字 58 が Hf の値）。そのため Z=72（Hf）では芯電荷 0・d 主量子数 0 となり、
   Hf を含む系は計算できない。ここでは Z<=86 の表を MOPAC（parameters_C.F90 の
   ios/iop/iod・eheat、mndod_C.F90 の iii/iiid、calpar.F90 の係数式）から作り直し、
   Z=57 以降を正しい原子番号の位置に置く（Z<=56 は PYSEQM の値をそのまま使う）。
2. PYSEQM の孤立原子エネルギーは s・p 項だけで、MOPAC の calpar の udd*iod と
   mndod.F90 の eiscor（d 殻の電子間反発）が入っていない。d 電子をもつ原子では
   生成熱が元素ごとに一定量ずれる（Zr で 412 kcal/mol）。eisol_d() がその差を返す。
"""
import json, os, math, torch
from ase.data import atomic_masses_iupac2016 as _MASS, chemical_symbols as _SYMS

NZ = 87                          # Z = 0..86
_T = json.load(open(os.path.join(os.path.dirname(__file__), "mopac_atom_tables.json")))
IOS, IOP, IOD = ([0] + _T[k] for k in ("ios", "iop", "iod"))   # 1 始まりにする
EHEAT = {int(k): v for k, v in _T["eheat"].items()}

def npq_s(z):   # mndod_C.F90: iii
    return 1 if z <= 2 else 2 if z <= 10 else 3 if z <= 18 else 4 if z <= 36 else 5 if z <= 54 else 6
def npq_d(z):   # mndod_C.F90: iiid（30*3, 18*4, 32*5, 6*6）
    return 3 if z <= 30 else 4 if z <= 48 else 5 if z <= 80 else 6

def _coef(z):
    s, k = IOS[z], IOP[z]; l = min(k, 6 - k)
    return dict(ussc=s, uppc=k, gssc=max(s - 1, 0), gspc=s * k, gp2c=(k * (k - 1)) // 2 + 0.5 * (l * (l - 1)) // 2,
                gppc=-0.5 * ((l * (l - 1)) // 2), hspc=-k * s * 0.5)

_patched = False
def patch_constants():
    global _patched
    if _patched: return
    from seqm.seqm_functions import constants as C
    orig = C.Constants.__init__
    def __init__(self, *a, **kw):
        orig(self, *a, **kw)
        f = float(self.eheat[1]) / 52.102            # kcal/mol → PYSEQM の内部単位
        def grow(name, fill):
            old = getattr(self, name).detach()
            new = torch.zeros(NZ, dtype=old.dtype); new[:57] = old[:57]
            for z in range(57, NZ): new[z] = fill(z)
            setattr(self, name, torch.nn.Parameter(new, requires_grad=False))
        grow("tore", lambda z: IOS[z] + IOP[z] + IOD[z])
        grow("atomic_num", lambda z: IOS[z] + IOP[z] + IOD[z])
        grow("qn", lambda z: float(npq_s(z))); grow("qn_int", lambda z: npq_s(z))
        grow("qnD_int", lambda z: npq_d(z) if (z == 57 or 71 <= z <= 79) else 0)
        for k in ("ussc", "uppc", "gssc", "gspc", "gp2c", "gppc", "hspc"):
            grow(k, lambda z, k=k: _coef(z)[k])
        grow("iso", lambda z: 0.0)
        grow("eheat", lambda z: EHEAT.get(z, 0.0) * f)
        grow("mass", lambda z: float(_MASS[z]))
        self.label = list(_SYMS[:NZ])
    C.Constants.__init__ = __init__
    _patched = True

def slater_condon(z, zsn, zpn, zdn):
    """MOPAC mndod.F90 scprm の r016, r066, r244, r266, r466（eV）"""
    from seqm.seqm_functions.two_elec_two_center_int import GetSlaterCondonParameter as rsc
    ns, nd = npq_s(z), npq_d(z)
    return dict(r016=rsc(0, ns, zsn, ns, zsn, nd, zdn, nd, zdn), r066=rsc(0, nd, zdn, nd, zdn, nd, zdn, nd, zdn),
                r244=rsc(2, ns, zsn, nd, zdn, ns, zsn, nd, zdn), r266=rsc(2, nd, zdn, nd, zdn, nd, zdn, nd, zdn),
                r466=rsc(4, nd, zdn, nd, zdn, nd, zdn, nd, zdn))

# mndod.F90 eiscor の整数表（Z: ir016, ir066, ir244, ir266, ir466）
_EIS = {}
for zs, cols in ((range(21, 30), ([2,4,6,5,10,12,14,16,10],[0,1,3,10,10,15,21,28,45],[1,2,3,5,5,6,7,8,5],[0,8,15,35,35,35,43,50,70],[0,1,8,35,35,35,36,43,70])),
                 (range(39, 48), ([2,4,4,5,10,7,8,0,10],[0,1,6,10,10,21,28,45,45],[1,2,4,5,5,5,5,0,5],[0,8,21,35,35,43,50,70,70],[0,1,21,35,35,36,43,70,70])),
                 (range(72, 81), ([4,6,5,10,12,14,9,10,0],[1,3,10,10,15,21,36,45,0],[2,3,5,5,6,7,5,5,0],[8,15,35,35,35,43,56,70,0],[1,8,35,35,35,36,56,70,0]))):
    for i, z in enumerate(zs): _EIS[z] = tuple(c[i] for c in cols)
_EIS[57] = _EIS[71] = (2, 0, 1, 0, 0)

def eisol_d(z, p):
    """MOPAC の EISOL と PYSEQM の Eiso の差（eV）= udd*iod + eiscor の項。
    p: その元素のパラメータ（U_dd, s/p/d_orb_exp_tail, F0SD, G2SD）を持つ dict"""
    if IOD[z] == 0 and z not in _EIS: return 0.0
    e = p["U_dd"] * IOD[z]
    if z in _EIS and p.get("d_orb_exp_tail", 0) > 0:
        r = slater_condon(z, p["s_orb_exp_tail"], p["p_orb_exp_tail"], p["d_orb_exp_tail"])
        if p.get("F0SD", 0) > 0.001: r["r016"] = p["F0SD"]
        if p.get("G2SD", 0) > 0.001: r["r244"] = p["G2SD"]
        a = _EIS[z]
        e += a[0]*r["r016"] + a[1]*r["r066"] - a[2]*r["r244"]/5 - a[3]*r["r266"]/49 - a[4]*r["r466"]/49
    return e

# ---- MOPAC parameters_C.F90 の main_group（.false. = 遷移金属・ランタノイド: 21-29, 39-47, 57-79）
def main_group(z):
    return not (21 <= z <= 29 or 39 <= z <= 47 or 57 <= z <= 79)

EV_CODATA = 27.211386245988
def _rsc(*a):
    """PYSEQM の GetSlaterCondonParameter（MOPAC rsc の移植）を CODATA の ev で評価する"""
    import seqm.seqm_functions.two_elec_two_center_int as T
    return T.GetSlaterCondonParameter(*a) * EV_CODATA / T.ev

def sp_two_electron(z, zsn, zpn):
    """MOPAC sp_two_electron.F90: 遷移金属では gss, gsp, hsp, gpp, gp2 をパラメータ表の値ではなく
    内部指数 zsn, zpn から計算した値で置き換える（calpar から呼ばれ、EISOL もこの値を使う）"""
    if zsn < 1e-4 or zpn < 1e-4 or main_group(z): return None
    ns = npq_s(z)
    r033 = _rsc(0, ns, zpn, ns, zpn, ns, zpn, ns, zpn); r233 = _rsc(2, ns, zpn, ns, zpn, ns, zpn, ns, zpn)
    return dict(gss=_rsc(0, ns, zsn, ns, zsn, ns, zsn, ns, zsn), gsp=_rsc(0, ns, zsn, ns, zsn, ns, zpn, ns, zpn),
                hsp=_rsc(1, ns, zsn, ns, zpn, ns, zsn, ns, zpn) / 3.0, gpp=r033 + 0.16 * r233, gp2=r033 - 0.08 * r233)

_rot_patched = False
def _replace_d(w, ni, nj, xij, rij, da, db, qa, qb, dpa, dpb, dsa, dsb, dda, ddb, r0a, r0b, r1a, r1b, r2a, r2b, rest):
    """d を含む二中心二電子積分を MOPAC rijkl の移植（dint）で計算し直す。引数は rotate に渡る値
    （PO2, PO3 は MOPAC の ddpo と同じ扱いにしたもの）"""
    from . import dint
    import seqm.seqm_functions.two_elec_two_center_int as T
    r3a, r3b, r4a, r4b, r5a, r5b, r6a, r6b = rest[:8]
    S2 = math.sqrt(2.0)
    PA = {"ddp": {2: da, 3: qa * S2, 4: dsa, 5: dpa, 6: dda}, "po": {2: r1a, 3: r2a, 4: r5a, 5: r4a, 6: r6a},
          "po1": r0a, "po7": r0a, "po8": r3a}
    PB = {"ddp": {2: db, 3: qb * S2, 4: dsb, 5: dpb, 6: ddb}, "po": {2: r1b, 3: r2b, 4: r5b, 5: r4b, 6: r6b},
          "po1": r0b, "po7": r0b, "po8": r3b}
    return dint.replace_d_block(w, ni, nj, xij, rij, PA, PB, float(T.ev))

def _pm7(w, out, ni, nj, rij, tore):
    """PM7 のとき d 成分の feathering と iod > 0 の平均化（dint.pm7_adjust）を施す"""
    from . import core, dint
    if core._active() != "PM7" or out[1] is None or out[1].dim() != 3 or out[1].shape[-1] != 9:
        return (w,) + tuple(out[1:])
    iod = torch.tensor(IOD[:NZ], device=ni.device)
    w, e1b, e2a = dint.pm7_adjust(w, out[1], out[2], ni, nj, rij, tore, iod[ni], iod[nj], core.feather)
    return (w, e1b, e2a) + tuple(out[3:])

def patch_rotate():
    """MOPAC と PYSEQM で二中心積分に使う Klopman–Ohno 項（PO）の扱いの違いを埋める。

    MOPAC（mndod.F90 の ddpo）は遷移金属の PO2（SP, L=1）と PO3（PP, L=2）を区間 [0.1, 5.0] bohr の
    黄金分割探索（poij）で求めるので 5.0 で頭打ちになる（例: Hf の PO2 = 5.0。上限なしの解は 5.716）。
    この値は芯–電子引力（e1b, e2a）と d 軌道を含む電子間積分に使われる。一方 sp 同士の電子間積分は
    calpar の値（上限なし）で計算される。HfH4 で H core（歪んだ構造）と Fock 行列の両方を
    MOPAC と照合して確かめた。PYSEQM はどちらにも同じ値を使うため、値が異なる遷移金属を含む
    組では rotate を 2 回呼び、電子間積分の sp–sp ブロック（45 個の対のうち先頭 10 個）だけを
    上限なしの値で計算したものに差し替える。

    典型元素: MOPAC の PO3 は 0.5/aq（hpp >= 0.1 に制限）。PYSEQM の rho2ad は制限なしの hpp で
    解くため、gpp < gp2 の元素（PM6 の F など）では 5.0 に張り付く → rho2（= MOPAC の PO3）に置き換える。"""
    global _rot_patched
    if _rot_patched: return
    import seqm.seqm_functions.two_elec_two_center_int as T
    orig = T.rotate
    T._pm7_orig_rotate = orig
    def tm(z):
        return ((z > 20) & (z < 30)) | ((z > 38) & (z < 48)) | ((z > 56) & (z < 80))
    def rotate(ni, nj, xij, rij, tore, da, db, qa, qb, dpa, dpb, dsa, dsb, dda, ddb, r0a, r0b, r1a, r1b, r2a, r2b, *rest, **kw):
        head = (ni, nj, xij, rij, tore, da, db, qa, qb, dpa, dpb, dsa, dsb, dda, ddb, r0a, r0b)
        mi, mj = tm(ni), tm(nj)
        rest = list(rest)       # rest[10], rest[11] = rho2ad, rho2bd（d 経路で使う PP, L=2）
        rest[10] = torch.where(mi, rest[10], r2a); rest[11] = torch.where(mj, rest[11], r2b)
        if not (mi.any() or mj.any()):
            out = orig(*head, r1a, r1b, r2a, r2b, *rest, **kw)
            if out[0].dim() != 3 or out[0].shape[-1] != 45: return out
            w = _replace_d(out[0], ni, nj, xij, rij, da, db, qa, qb, dpa, dpb, dsa, dsb, dda, ddb, r0a, r0b, r1a, r1b, r2a, r2b, rest)
            return _pm7(w, out, ni, nj, rij, tore)
        # ddpo の値（芯–電子引力、d を含む積分）
        c1a = torch.where(mi, r1a.clamp(0.1, 5.0), r1a); c1b = torch.where(mj, r1b.clamp(0.1, 5.0), r1b)
        rB = list(rest); rB[10] = torch.where(mi, rest[10].clamp(0.1, 5.0), rest[10]); rB[11] = torch.where(mj, rest[11].clamp(0.1, 5.0), rest[11])
        c2a = torch.where(mi, rB[10], r2a); c2b = torch.where(mj, rB[11], r2b)
        outB = orig(*head, c1a, c1b, c2a, c2b, *rB, **kw)
        same = all(torch.equal(u, v) for u, v in ((c1a, r1a), (c1b, r1b), (c2a, r2a), (c2b, r2b)))
        if outB[0].dim() != 3 or outB[0].shape[-1] != 45:
            return outB
        if same:
            w = outB[0]
        else:
            # calpar の値（sp–sp の電子間積分）
            outA = orig(*head, r1a, r1b, r2a, r2b, *rest, **kw)
            w = outB[0].clone(); w[:, :10, :10] = outA[0][:, :10, :10]
        w = _replace_d(w, ni, nj, xij, rij, da, db, qa, qb, dpa, dpb, dsa, dsb, dda, ddb, r0a, r0b, c1a, c1b, c2a, c2b, rB)
        return _pm7(w, outB, ni, nj, rij, tore)
    T.rotate = rotate
    _rot_patched = True
