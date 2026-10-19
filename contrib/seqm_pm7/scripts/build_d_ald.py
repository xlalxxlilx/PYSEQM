import numpy as np, os, subprocess, re
from ase import Atoms
import ase.io
XTB=os.environ.get("XTB", "xtb")   # xtb の実行ファイル
T4=np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]])/np.sqrt(3)
def perp(u):
    a=np.cross(u,[0.37,0.21,0.9]); a/=np.linalg.norm(a); return a, np.cross(u,a)
def methyl(c, away):
    u=away/np.linalg.norm(away); a,b=perp(u)
    return [c+0.36*u+1.03*(np.cos(t)*a+np.sin(t)*b) for t in (0,2*np.pi/3,4*np.pi/3)]
def nme2(M, N):
    s=[];X=[]
    u=(N-M)/np.linalg.norm(N-M); a,_=perp(u)
    for sg in (1,-1):
        C=N+1.46*(0.5*u+sg*0.866*a); s.append("C"); X.append(C)
        for h in methyl(C, C-N): s.append("H"); X.append(h)
    return s,X
def amide_complex(Mel, dirs, rMN=2.07):
    s=[Mel]; X=[np.zeros(3)]
    for d in dirs:
        N=rMN*d/np.linalg.norm(d); s.append("N"); X.append(N)
        ss,xx=nme2(np.zeros(3),N); s+=ss; X+=xx
    return s,np.array(X)
M={}
M["TDMAZ"]=amide_complex("Zr",T4); M["TDMAH"]=amide_complex("Hf",T4,2.05)
# CpZr(NMe2)3
s,X=amide_complex("Zr",[np.array([np.cos(t),np.sin(t),-0.45]) for t in (0,2*np.pi/3,4*np.pi/3)])
for k in range(5):
    t=2*np.pi*k/5; c=np.array([1.2*np.cos(t),1.2*np.sin(t),2.25]); s.append("C"); X=np.vstack([X,c])
    s.append("H"); X=np.vstack([X,c+np.array([1.08*np.cos(t),1.08*np.sin(t),0.1])])
M["CpZrNMe23"]=(s,X)
# SiH2(NMe2)2
s=["Si"];X=[np.zeros(3)]
for d in T4[:2]:
    N=1.72*d; s.append("N"); X.append(N); ss,xx=nme2(np.zeros(3),N); s+=ss; X+=xx
for d in T4[2:]: s.append("H"); X.append(1.48*d)
M["SiH2NMe22"]=(s,np.array(X))
M["HfCl4"]=(["Hf"]+["Cl"]*4, np.vstack([[0,0,0],2.33*T4]))
# TMA·H2O 付加体
s,X=["Al"],[np.zeros(3)]
for d in (np.array([1,0,-0.3]),np.array([-0.5,0.866,-0.3]),np.array([-0.5,-0.866,-0.3])):
    C=1.97*d/np.linalg.norm(d); s.append("C"); X.append(C); 
    for h in methyl(C,C): s.append("H"); X.append(h)
s+=["O","H","H"]; X+=[np.array([0,0,2.0]),np.array([0.78,0,2.55]),np.array([-0.78,0,2.55])]
M["TMA_H2O"]=(s,np.array(X))
# アダマンタン型 M4O6(OH)4 と Al4O6
def adam(Mel,a,b,rOH=None,rMO=1.95):
    Mv=a*T4*np.sqrt(3); s=[Mel]*4; X=list(Mv)
    for i in range(4):
        for j in range(i+1,4):
            m=(Mv[i]+Mv[j])/2; s.append("O"); X.append(m/np.linalg.norm(m)*b)
    if rOH:
        for i in range(4):
            u=Mv[i]/np.linalg.norm(Mv[i]); O=Mv[i]+rMO*u; s.append("O"); X.append(O)
            a_,_=perp(u); s.append("H"); X.append(O+0.96*(0.6*u+0.8*a_))
    return s,np.array(X)
M["Zr4O6OH4"]=adam("Zr",1.3,2.3,True,1.97); M["Hf4O6OH4"]=adam("Hf",1.3,2.3,True,1.95)
M["Si4O6OH4"]=adam("Si",1.05,1.9,True,1.63); M["Al4O6"]=adam("Al",1.15,2.0)
import sys
only=sys.argv[1:] 
for nm,(s,x) in M.items():
    if only and nm not in only: continue
    d=f"mols2/{nm}"; os.makedirs(d,exist_ok=True)
    ase.io.write(f"{d}/in.xyz", Atoms(s,positions=x))
    r=subprocess.run([XTB,"in.xyz","--gfn","2","--opt"],cwd=d,capture_output=True,text=True,env={**os.environ,"OMP_NUM_THREADS":"4"})
    ok=os.path.exists(f"{d}/xtbopt.xyz"); print(nm,len(s),"ok" if ok else "FAIL", re.findall(r"TOTAL ENERGY\s+([-0-9.]+)",r.stdout)[-1:],flush=True)
