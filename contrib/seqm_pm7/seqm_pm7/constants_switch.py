"""PYSEQM の物理定数（ev=27.21, a0=0.529167 の旧値）を CODATA（MOPAC 23 の既定）に切り替える。
PYSEQM は ev / a0 を各モジュールで `from .constants import ev` として取り込んでいるので、
読み込み済みの seqm.* モジュールの属性を書き換える。関数内の参照は呼び出し時に解決されるので反映される。"""
import sys
OLD = {"ev": 27.21, "a0": 0.529167}
CODATA = {"ev": 27.211386245988, "a0": 0.529177210903}
_current = "old"

def use(which="codata"):
    global _current
    import seqm, seqm.basics, seqm.seqm_functions.energy, seqm.seqm_functions.two_elec_two_center_int  # 読み込みを確定
    val = CODATA if which == "codata" else OLD
    for name, mod in list(sys.modules.items()):
        if not name.startswith("seqm") or mod is None: continue
        for k in ("ev", "a0"):
            if hasattr(mod, k) and isinstance(getattr(mod, k), float):
                setattr(mod, k, val[k])
    from . import core
    core._EV = val["ev"]; core._A0 = val["a0"]
    _current = which
    return val

def current(): return _current
