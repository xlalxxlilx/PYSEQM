"""ASE 用の計算機（PM7 / PM7- / PM6、MOPAC 23.2.5 互換）"""
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from .calculator import compute, Topology
KCAL2EV = 1.0 / 23.060547830619029

class SeqmPM7(Calculator):
    """energy は生成熱（eV 換算）。freeze_topology=True なら MOPAC と同じく最初の構造の結合リストを使い続ける。"""
    implemented_properties = ["energy", "forces"]
    def __init__(self, method="PM7", charge=0, device="cpu", freeze_topology=True, reuse_density=True, **kw):
        """reuse_density=True: 前の構造で収束した密度行列を次の SCF の初期値にする（構造最適化・MD で速くなる）"""
        super().__init__(**kw); self.method = method; self.chg = charge; self.device = device
        self.freeze = freeze_topology; self.topo = None; self.last = None
        self.reuse = reuse_density; self.P = None
    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        if self.topo is None or not self.freeze or "numbers" in system_changes:
            self.topo = Topology(atoms.numbers, atoms.positions); self.P = None
        P0 = self.P if (self.reuse and self.P is not None) else None
        r = compute(list(atoms.numbers), atoms.positions, self.method, charge=self.chg, device=self.device, topology=self.topo,
                    P0=P0, return_density=self.reuse)
        if self.reuse and r.get("scf_converged", True): self.P = r["P"].detach()
        self.last = r
        self.results = {"energy": float(r["Hf"]) * KCAL2EV, "forces": r["forces"] * KCAL2EV}
