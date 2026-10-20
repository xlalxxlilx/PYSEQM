import sys,os,time,json,itertools,numpy as np,torch
W=os.path.join(os.path.dirname(os.path.abspath(__file__)),".."); sys.path.insert(0,W)
os.environ.setdefault("SEQM_PM7_OZ2_LIB", W+"/ozaki/liboz2shim.so")
import seqm_pm7.calculator as C, seqm_pm7.scf_fast as SF, seqm_pm7.eig_mixed as EM
R=json.load(open(W+"/validation/mopac_ref.json")); w=R["cl:HF_w20_start_neutral"]
offs=[np.array(v)*12.0 for v in itertools.product(range(3),repeat=3)]; s=w["sym"]*27; x=np.vstack([np.array(w["x"])+o for o in offs])
res={}
for eig,passes in (("fp64",1),("auto",1),("auto",2),("fp64",1),("auto",1),("auto",2)):
    EM.PASSES=passes; SF.STATS["fock"]=0; torch.cuda.synchronize(); t=time.time()
    q=C.compute(s,x,"PM7",device="cuda",cutoff=10.0,eig=eig); torch.cuda.synchronize(); dt=time.time()-t
    key=(eig,passes); res.setdefault(key,[]).append((dt,SF.STATS["fock"],float(q["Hf"]),q["forces"]))
    print(f"eig={eig} passes={passes} {dt:5.1f}s fock {SF.STATS['fock']} Hf {q['Hf']:.6f}",flush=True)
ref=res[("fp64",1)][0]
for k,v in res.items():
    for dt,f,H,F in v: print(k, f"dHf {H-ref[2]:+.1e} dF {np.abs(F-ref[3]).max():.1e}")
