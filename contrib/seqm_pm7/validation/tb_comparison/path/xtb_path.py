import os, sys, json, subprocess, re, ase.io, concurrent.futures as cf
W = os.path.abspath("../../pyseqm_test/scans")
def run(job):
    tag, i, gfn, a = job
    d = f"w/{tag}_{gfn}_{i}"; os.makedirs(d, exist_ok=True)
    ase.io.write(f"{d}/m.xyz", a)
    r = subprocess.run(["xtb", "m.xyz", "--gfn", str(gfn), "--sp", "--acc", "0.1"], cwd=d, capture_output=True, text=True,
                       env={**os.environ, "OMP_NUM_THREADS": "1", "OMP_STACKSIZE": "1G"})
    E = float(re.findall(r"TOTAL ENERGY\s+(-?\d+\.\d+)", r.stdout)[-1]) * 27.211386245988
    return tag, i, gfn, E
jobs = []
for tag in ("HF_w4", "HF_w8", "NH3_w4", "NH3_w8"):
    fr = ase.io.read(f"{W}/{tag}_UMA_scan.xyz", index=":") + [ase.io.read(f"{W}/{tag}_UMA_molopt.xyz"), ase.io.read(f"{W}/{tag}_UMA_release.xyz")]
    for gfn in (1, 2):
        for i, a in enumerate(fr): jobs.append((tag, i, gfn, a))
out = {}
with cf.ProcessPoolExecutor(8) as ex:
    for tag, i, gfn, E in ex.map(run, jobs):
        out.setdefault(tag, {}).setdefault(f"GFN{gfn}", [None] * 21)[i] = E
json.dump(out, open("xtb_on_path.json", "w"), indent=1)
print({t: {m: len(v) for m, v in d.items()} for t, d in out.items()})
