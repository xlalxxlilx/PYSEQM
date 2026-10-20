import time, json, torch, oz2
d = "cuda"; torch.manual_seed(0)
def tm(f, rep=5):
    f(); torch.cuda.synchronize(); t = time.time()
    for _ in range(rep): f()
    torch.cuda.synchronize(); return (time.time() - t) / rep
rows = []
for n in (324, 1000, 2000, 3375, 4096):
    # Fock 行列（−40〜15 eV の対称行列）× 固有ベクトル（直交行列）の積を模す
    Q, _ = torch.linalg.qr(torch.randn(n, n, dtype=torch.float64, device=d))
    lam = torch.linspace(-40, 15, n, dtype=torch.float64, device=d)
    F = (Q * lam) @ Q.T
    ref = F @ Q
    tn = tm(lambda: F @ Q)
    print(f"n={n:5d} native DGEMM {tn*1e3:8.2f} ms ({2*n**3/tn/1e12:.2f} TFLOPS)", flush=True)
    for be, mods in (("int8", (13, 14, 15, 16)), ("fp8", (10, 11, 12))):
        for m in mods:
            try:
                C = oz2.matmul(F, Q, moduli=m, backend=be)
                err = float((C - ref).abs().max() / ref.abs().max())
                t = tm(lambda: oz2.matmul(F, Q, moduli=m, backend=be))
                rows.append(dict(n=n, backend=be, moduli=m, t_ms=t*1e3, t_native_ms=tn*1e3, speedup=tn/t, relerr=err))
                print(f"        {be} moduli {m:2d}: {t*1e3:8.2f} ms  x{tn/t:5.2f}  rel.err {err:.1e}", flush=True)
            except Exception as e:
                print(f"        {be} moduli {m}: ERR {e}", flush=True)
    print(f"        (native の誤差の目安: {float(((F @ Q) - ref).abs().max()):.1e})")
json.dump(rows, open("bench_oz2.json", "w"), indent=1)
