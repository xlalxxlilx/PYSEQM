"""MOPAC の parameters_for_PM7_C.F90 から PYSEQM 形式のパラメータを作る（パッケージ同梱用）"""
import os
from .convert import parse, write
from .tables import sp_two_electron

def _tm_sp(el):
    """遷移金属の gss…gp2 を MOPAC と同じく zsn, zpn から計算し直す"""
    for z, d in el.items():
        r = sp_two_electron(z, d.get(("zsn", None), 0.0), d.get(("zpn", None), 0.0))
        if r:
            for k, v in r.items(): d[(k, None)] = v
HERE = os.path.dirname(os.path.abspath(__file__))
PM7_DIR = os.path.join(HERE, "params_pm7") + "/"
PM6_DIR = os.path.join(HERE, "params_pm6") + "/"

def build(force=False):
    if os.path.exists(PM7_DIR + "parameters_PM6_MOPAC.csv") and not force: return PM7_DIR
    os.makedirs(PM7_DIR, exist_ok=True)
    el, pairs, vpar = parse("parameters_for_PM7_C.F90", "7"); _tm_sp(el)
    # MOPAC ccrep（PM7）: 二原子パラメータが無い組は同核の値の平均で埋める
    for i in sorted(el):
        for j in sorted(el):
            if j > i or i >= 99: continue
            if abs(pairs.get((i, j), {}).get("xfac", 0.0)) < 1e-5:
                pi, pj = pairs.get((i, i), {}), pairs.get((j, j), {})
                pairs[(i, j)] = {"xfac": 0.5*(pi.get("xfac", 0)+pj.get("xfac", 0)), "alpb": 0.5*(pi.get("alpb", 0)+pj.get("alpb", 0))}
    # PYSEQM の sp 版（PM6_SP）と spd 版（PM6）の両方のファイル名で書く
    write(el, pairs, "PM6_SP", PM7_DIR)
    write(el, pairs, "PM6", PM7_DIR)      # d 軌道をもつ系は PYSEQM の "PM6"（spd）経路で読む
    d_elems = sorted(z for z, d in el.items() if abs(d.get(("zd", None), 0.0)) > 0)
    with open(PM7_DIR + "d_elements.txt", "w") as f: f.write(" ".join(map(str, d_elems)))
    with open(PM7_DIR + "v_par.txt", "w") as f: f.write(repr(vpar))
    return PM7_DIR


def build_pm6(force=False):
    """PYSEQM 同梱の PM6 は 5 桁に丸めてあるので、MOPAC の parameters_for_PM6_C.F90 から全桁で作る（sp 元素用）"""
    if os.path.exists(PM6_DIR + "parameters_PM6_MOPAC.csv") and not force: return PM6_DIR
    os.makedirs(PM6_DIR, exist_ok=True)
    el, pairs, vpar = parse("parameters_for_PM6_C.F90", "6"); _tm_sp(el)
    write(el, pairs, "PM6_SP", PM6_DIR)
    write(el, pairs, "PM6", PM6_DIR)
    d_elems = sorted(z for z, d in el.items() if abs(d.get(("zd", None), 0.0)) > 0)
    with open(PM6_DIR + "d_elements.txt", "w") as f: f.write(" ".join(map(str, d_elems)))
    return PM6_DIR
