import sys,os,time,json,itertools,numpy as np,ase.io,torch; sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),".."))
import seqm_pm7.calculator as C
def load(nm): a=ase.io.read(os.path.join(os.environ.get("SEQM_PM7_MOLS",os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","validation","mols2")),nm,"xtbopt.xyz")); return a.get_chemical_symbols(), a.positions-a.positions.mean(0)
def grid(s,x,n,d): offs=[np.array(v)*d for v in itertools.product(range(n),repeat=3)]; return s*len(offs), np.vstack([x+o for o in offs])
R=json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","validation","mopac_ref.json"))); w=R["cl:HF_w20_start_neutral"]
s1,x1=load("TDMAZ"); s2,x2=load("Zr4O6OH4")
sets={"27x Zr4O6(OH)4 (486)":grid(s2,x2,3,9.0),"27x TDMAZ (999)":grid(s1,x1,3,11.0),"27x HF(H2O)20 (1674)":grid(w["sym"],np.array(w["x"]),3,12.0)}
out=[]
for nm,(s,x) in sets.items():
    res={}
    for rc in (None,10.0):
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        try:
            t=time.time(); q=C.compute(s,x,"PM7",device="cuda",cutoff=rc); torch.cuda.synchronize(); dt=time.time()-t
            res[rc]=q; mem=torch.cuda.max_memory_allocated()/2**30
            print(f"{nm:22s} Rc {rc} {dt:7.2f}s peakGPU {mem:5.1f} GiB conv {q['scf_converged']}",flush=True)
            out.append(dict(system=nm,cutoff=rc,t=dt,peak_gib=mem))
        except torch.cuda.OutOfMemoryError as e:
            print(f"{nm:22s} Rc {rc} OOM",flush=True); out.append(dict(system=nm,cutoff=rc,t=None,peak_gib=None,oom=True))
    if None in res and 10.0 in res:
        print(f"   dHf {res[10.0]['Hf']-res[None]['Hf']:+.2e} dF {np.abs(res[10.0]['forces']-res[None]['forces']).max():.2e}",flush=True)
        out[-1].update(dHf=float(res[10.0]['Hf']-res[None]['Hf']), dF=float(np.abs(res[10.0]['forces']-res[None]['forces']).max()))
json.dump(out,open("big_test_cuda.json","w"),indent=1)
