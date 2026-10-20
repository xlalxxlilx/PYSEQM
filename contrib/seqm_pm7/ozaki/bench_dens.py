import time, torch, oz2, json
from dens_mixed import density_mixed
d = "cuda"; torch.manual_seed(0)
def tm(f, rep=3):
    f(); torch.cuda.synchronize(); t = time.time()
    for _ in range(rep): r = f()
    torch.cuda.synchronize(); return (time.time() - t) / rep, r
rows = []
for n in (500, 1000, 1500, 2000, 3375, 4096):
    nocc = n // 2
    Q, _ = torch.linalg.qr(torch.randn(n, n, dtype=torch.float64, device=d))
    # Fock らしい固有値（占有 −40〜−11 eV、非占有 1〜15 eV、ほぼ縮退を含む）
    lam = torch.cat([torch.linspace(-40, -11, nocc, device=d, dtype=torch.float64), torch.linspace(1, 15, n - nocc, device=d, dtype=torch.float64)])
    F = (Q * lam) @ Q.T; F = (F + F.T) / 2
    t64, (e, V) = tm(lambda: torch.linalg.eigh(F)); P64 = 2 * V[:, :nocc] @ V[:, :nocc].T
    tmix, P = tm(lambda: density_mixed(F, nocc))
    tnat, Pn = tm(lambda: density_mixed(F, nocc, mm=lambda a, b: a @ b))
    err = float((P - P64).abs().max()); errn = float((Pn - P64).abs().max())
    idem = float((P @ P / 2 - P).abs().max())
    rows.append(dict(n=n, t_eigh64_ms=t64*1e3, t_mixed_oz2_ms=tmix*1e3, t_mixed_native_ms=tnat*1e3, err=err, err_native=errn, idempotency=idem))
    print(f"n={n:5d} FP64 eigh {t64*1e3:7.1f} ms | 単精度+補正（尾崎 II）{tmix*1e3:7.1f} ms x{t64/tmix:4.2f} 誤差 {err:.1e} 冪等性 {idem:.1e} | 補正をネイティブ GEMM {tnat*1e3:7.1f} ms 誤差 {errn:.1e}", flush=True)
json.dump(rows, open("bench_dens.json", "w"), indent=1)
