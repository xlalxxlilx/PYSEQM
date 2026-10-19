"""PYSEQM の GenerateRotationMatrix（RotationMatrixD.py）の自動微分を速くする

元の関数は (原子対数, 15, 45) の行列に要素ごとの in-place 代入を約 1000 回行う。
自動微分の逆伝播では代入 1 回ごとに行列全体が複製されるため、原子対が多いと極端に遅い
（148 原子・約 1.1 万対で逆伝播が約 16 s）。

ここでは元のソースを AST で機械的に変換し、`matrix[..., a, b] = 式` を「式の値を記録」に置き換え、
最後に 1 回の index_put で行列を作る。式と代入の順序は元のまま（同じ要素への再代入は最後の値を採る）。
"""
import ast, inspect, textwrap
import torch
import seqm.seqm_functions.RotationMatrixD as RD
import seqm.seqm_functions.two_elec_two_center_int as TT

_ORIG = RD.GenerateRotationMatrix


def _build(matrix, rec):
    n, R, Cn = matrix.shape
    last = {}
    for a, b, v in rec:
        last[(int(a), int(b))] = v
    keys = list(last)
    vals = []
    for k in keys:
        v = last[k]
        vals.append(v if torch.is_tensor(v) else torch.full((n,), float(v), dtype=matrix.dtype, device=matrix.device))
    V = torch.stack([v.expand(n) for v in vals], dim=1)                       # (n, nent)
    lin = torch.tensor([a * Cn + b for a, b in keys], device=matrix.device)
    flat = matrix.new_zeros(n, R * Cn)
    flat = flat.index_copy(1, lin, V)                                          # 関数形（in-place ではない）
    return flat.view(n, R, Cn)


class _T(ast.NodeTransformer):
    def visit_Assign(self, node):
        t = node.targets[0]
        if (len(node.targets) == 1 and isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name)
                and t.value.id == "matrix" and isinstance(t.slice, ast.Tuple) and len(t.slice.elts) == 3
                and isinstance(t.slice.elts[0], ast.Constant) and t.slice.elts[0].value is Ellipsis):
            call = ast.Call(func=ast.Attribute(value=ast.Name("_REC", ast.Load()), attr="append", ctx=ast.Load()),
                            args=[ast.Tuple([t.slice.elts[1], t.slice.elts[2], node.value], ast.Load())], keywords=[])
            return ast.copy_location(ast.Expr(call), node)
        return node

    def visit_Return(self, node):
        new = ast.Return(ast.Call(func=ast.Name("_build", ast.Load()), args=[node.value, ast.Name("_REC", ast.Load())], keywords=[]))
        return ast.copy_location(new, node)


def _make():
    src = textwrap.dedent(inspect.getsource(_ORIG))
    tree = ast.parse(src)
    fn = tree.body[0]
    fn.name = "GenerateRotationMatrix_fast"
    fn = _T().visit(fn)
    fn.body.insert(0, ast.parse("_REC = []").body[0])
    tree.body[0] = fn
    ast.fix_missing_locations(tree)
    g = dict(RD.__dict__); g["_build"] = _build
    exec(compile(tree, "<rot_fast>", "exec"), g)
    return g["GenerateRotationMatrix_fast"]


GenerateRotationMatrix_fast = _make()


def patch():
    TT.GenerateRotationMatrix = GenerateRotationMatrix_fast


# --- Rotate2Center2Electron: YM の要素ごとの代入（約 1000 回）を 1 回の index_copy に ---------------
_ORIG_R2C = RD.Rotate2Center2Electron


def _ym_index():
    """元の while ループと同じ順序で (YM の行, 回転行列の行, 列) を列挙する"""
    src = inspect.getsource(_ORIG_R2C)
    g = {"np": RD.np}
    # MET / META / METB / METI の定義部分だけを取り出して評価する
    body = src.split("i = 0", 1)[0].split('"""', 2)[-1] if '"""' in src else src
    lines = textwrap.dedent("\n".join(src.split("\n")[1:])).split("YM = torch.zeros_like(WW)")[0]
    exec(lines, g)
    MET, META, METB, METI = g["MET"], g["META"], g["METB"], g["METI"]
    rows, Is, cols = [], [], []
    KL = 1
    while KL < 45:
        mkl = MET[KL] - 1
        NKL = METB[mkl] - META[mkl] + 1
        for I in range(NKL):
            rows.append(int(METI[I, mkl] - 1)); Is.append(I); cols.append(KL)
        KL += 1
    last = {}
    for r, I, c in zip(rows, Is, cols):
        last[(r, c)] = I                                   # 同じ要素への再代入は最後の値
    return [k[0] for k in last], [v for v in last.values()], [k[1] for k in last]


_YR, _YI, _YC = _ym_index()


def Rotate2Center2Electron_fast(WW, rotationMatrix):
    n = WW.shape[0]
    dev = WW.device
    r = torch.tensor(_YR, device=dev); I = torch.tensor(_YI, device=dev); c = torch.tensor(_YC, device=dev)
    vals = rotationMatrix[:, I, c]                                                  # (n, nent)
    flat = WW.new_zeros(n, 45 * 45)
    lin = torch.cat([torch.zeros(1, dtype=torch.long, device=dev), r * 45 + c])
    vals = torch.cat([torch.ones(n, 1, dtype=WW.dtype, device=dev), vals], dim=1)   # YM[0,0] = 1
    YM = flat.index_copy(1, lin, vals).view(n, 45, 45)
    STEP1 = torch.bmm(YM, WW.transpose(1, 2))
    FIN = torch.bmm(STEP1, YM.transpose(1, 2))
    FIN = torch.cat([torch.cat([WW[:, :10, :10], FIN[:, :10, 10:]], dim=2), FIN[:, 10:, :]], dim=1)
    return FIN.transpose(1, 2)


def patch():
    TT.GenerateRotationMatrix = GenerateRotationMatrix_fast
    TT.Rotate2Center2Electron = Rotate2Center2Electron_fast
