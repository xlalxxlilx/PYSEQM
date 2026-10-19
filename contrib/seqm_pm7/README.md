# seqm_pm7 v0.2 — PYSEQM 用の PM7（MOPAC 23.2.5 互換）と PM6 の互換補正（d 元素対応）

> **English summary.** `seqm_pm7` adds MOPAC-23.2.5-compatible **PM7**, **PM7-** and **PM6** (with MOPAC's
> corrections) to PYSEQM, including elements with d orbitals (validated for H, C, N, O, F, Al, Si, Cl, Zr, Hf).
> It does not modify PYSEQM's files; it patches functions at import time. Heats of formation agree with
> MOPAC 23.2.5 to ≤ 0.004 kcal/mol on 22 d-element molecules and 106 sp molecules. The patches fix several
> issues in PYSEQM 2.0.0's d-orbital path (atom tables for Z ≥ 57, missing d terms in the isolated-atom energy,
> STO overlap expansions and d–d rotation, some (dd|pp) two-centre integrals, PO3 for F, SCF convergence).
> Licensed under Apache-2.0 (derived from OpenMOPAC); see `LICENSE` and `NOTICE`. Documentation below is in Japanese.


PYSEQM（LANL、PyTorch 実装の NDDO 法）に、MOPAC の **PM7** と、MOPAC と同じ補正つきの **PM6** を外付けで追加するパッケージ。
PYSEQM 本体のファイルは書き換えず、読み込み時に関数を差し替える。

## 何が入っているか

| 項目 | 内容 | MOPAC の対応箇所 |
|---|---|---|
| ① パラメータ | `parameters_for_PM7_C.F90` / `parameters_for_PM6_C.F90` から全桁で変換（PYSEQM 同梱の PM6 は 5 桁丸め）。PM7 で二原子パラメータが無い組は同核の平均で埋める | `models/`, `integrals/ccrep.F90` |
| ② 核間反発 | `ccrep` の PM6/PM7 分岐を忠実に移植（C–C 項、O–H 追加項、Si–O 項、VdW Gaussian、非分極芯項） | `integrals/ccrep.F90` |
| ②' feathering（PM7 のみ） | 二中心積分・電子–芯引力・核間反発の γ を r → 7 Å で点電荷に切り替える（**文書化されていない PM7 固有処理**。これが無いと大きな系で 1 kcal/mol 以上ずれる） | `integrals/mndod.F90` の `to_point`、`input/readmo.F90` の `l_feather = method_PM7` |
| ③ MM 補正 | C≡C 三重結合（PM6/PM7）、アミド NHCO 二面角（PM6 2.5 / PM7 3.1595 kcal/mol）、3 結合 N のピラミッド化（PM6 のみ） | `corrections/set_up_dentate.F90`, `moldat.F90` |
| ④ 分散 | PM6-DH 型（PM7 の α=15.450118, s=1.226593, スケール 2.286419） | `corrections/H_bond_correction_PM6_DH_Dispersion.F90` |
| ⑤ 水素結合 | EH_plus 型（供与体・受容体の判定、角度・二面角、減衰、O–O 短距離項） | `corrections/Hydrogen_bond_corrections.F90`, `H_bond_correction_EH_plus.F90`, `H_bond_correction_bits.F90` |
| ⑥ d 元素（v0.2） | Al, Si, Zr, Hf など d 軌道をもつ元素を PYSEQM の spd 経路で計算し、MOPAC と一致させるための修正一式（下の「d 元素への対応」） | `integrals/mndod.F90`, `models/calpar.F90`, `models/parameters_C.F90` |
| ⑦ Si–O–H 補正（PM7） | Si と H の両方に結合した O ごとの角度補正 15·(θ − 125°)²·exp(…) | `corrections/set_up_dentate.F90` の `Si_O_H_Correction` |
| 物理定数 | 既定で CODATA（MOPAC 23 の既定）。`constants="old"` で PYSEQM・MOPAC OLDFPC の旧値 | `conref_C.F90` |

補正項はすべて torch で書いてあり、力は自動微分で得る（MOPAC は分散・水素結合の勾配を前進差分で近似している）。

## 使い方

```python
import seqm_pm7
r = seqm_pm7.compute(symbols, coords, "PM7", charge=0)   # coords: (n,3) Å または (nmol,n,3)
r["Hf"]        # 生成熱 kcal/mol（MOPAC の FINAL HEAT OF FORMATION に相当）
r["forces"]    # kcal/mol/Å
r["E_disp"], r["E_hb"], r["E_CC"], r["E_NHCO"]   # 補正項の内訳 kcal/mol
# method: "PM7"（MOPAC PM7）, "PM7-"（補正なし = MOPAC PM7-）, "PM6"（MOPAC PM6）

from seqm_pm7 import SeqmPM7          # ASE 計算機（energy は Hf を eV に換算）
atoms.calc = SeqmPM7("PM7", charge=0)
```

環境: PYSEQM 2.0.0、torch 2.6 で確認。`PYTHONPATH` にこのディレクトリ（`contrib/seqm_pm7`）を入れる。
照合用のスクリプトは環境変数 `MOPAC_HOME`（MOPAC 23.2.5 の展開先、既定 `~/opt/mopac-23.2.5-linux`）、`MOPAC_SRC`（MOPAC のソース、既定 `~/src/mopac`、パラメータの再変換時のみ）、`XTB`（xtb の実行ファイル）を見る。
パラメータ変換には MOPAC のソース（`~/src/mopac`、v23.2.5 と同一内容の main）が要るが、変換済みの CSV を同梱してある。

## d 元素への対応（v0.2）

PYSEQM 2.0.0 の d 経路（`method="PM6"`）には、MOPAC と結果が合わない原因が多数あった。PYSEQM のファイルは書き換えず、すべて読み込み時の差し替えで直している。

| # | PYSEQM 側の問題 | 影響 | 対処（モジュール） |
|---|---|---|---|
| 1 | 原子定数表がランタノイドを飛ばした並びで、添字 57 以降が原子番号とずれている（添字 58 が Hf の値） | Hf（Z=72）の芯電荷 0・d 主量子数 0 で計算が破綻 | Z≤86 の表を MOPAC の ios/iop/iod・eheat・iii/iiid から作り直す（`tables.patch_constants`） |
| 2 | 孤立原子エネルギーが s・p 項だけ（CSV の EISOL 列は未使用） | d 電子をもつ原子ごとに生成熱が一定量ずれる（Zr −412.3、Hf −458.4 kcal/mol） | MOPAC の udd·iod と `eiscor`（Slater–Condon 積分）を移植（`tables.eisol_d`） |
| 3 | 遷移金属の gss・gsp・hsp・gpp・gp2 をパラメータ表の値のまま使う | MOPAC は内部指数 zsn, zpn から計算し直す（`sp_two_electron`） | パラメータ変換時に計算し直す（`tables.sp_two_electron`） |
| 4 | 重なり積分の展開式が (n, l) の組ごとの手書きで、5d–1s、4d–4d、5p–3p、6s–3s などで誤り（例: Zr–Zr の dz2–s が 1.64） | Hf の系で数十 kcal/mol、Zr 二核で 4 kcal/mol | 一般の Slater 型軌道の重なり（回転楕円体座標の補助積分 A_k, B_k）を新規実装（`sto_overlap`） |
| 5 | d–d の組の重なりの回転（局所系 → 分子系）が向きによって誤る | Cl2・SiCl4 などのエネルギーが分子の向きで最大 0.24 kcal/mol 変わる | 実球面調和関数の回転を自前で行う（`sto_overlap.overlap_molecular`、pyscf の数値格子積分と 1e-12 で一致） |
| 6 | d を含む二中心二電子積分の手書きの式の一部（d 原子の dd 分布 × 典型元素の pp 分布）が誤り | M(OH)n で +0.03〜0.07 kcal/mol、Al2O3H2 の力で 0.5 kcal/mol/Å | MOPAC の `rijkl` / `charg`（多重極展開）を移植して d を含む成分を置き換える（`dint`） |
| 7 | 典型元素の PP, L=2 の Klopman–Ohno 項を hpp の下限なしで解く | gpp < gp2 の元素（PM6 の F）で 5.0 bohr に張り付き、F 1 個あたり +0.7 kcal/mol | MOPAC の PO3（hpp ≥ 0.1）に置き換える（`tables.patch_rotate`） |
| 8 | 遷移金属の PO2・PO3 に上限なしの値を使う | MOPAC は芯–電子引力と d を含む積分に [0.1, 5.0] bohr の探索値（Hf の PO2 = 5.0）、sp 同士の電子間積分に上限なしの値（5.716）を使う | 2 通りの値で積分を作り分ける（`tables.patch_rotate`） |
| 9 | PM7 の d 固有処理が無い | Zr で数十〜百数十 kcal/mol | d 成分の feathering、iod > 0 の原子（Zr, Hf など）の d–p・d–s クーロン積分と芯引力の平均化（`dint.pm7_adjust`、`mndod.F90` の rotatd の `method_PM7` ブロック）、Si–O–H 補正 |
| 10 | Pulay DIIS が誤った SCF 解に落ちる（PM6 の ZrH4 で +108 kcal/mol）、適応混合は収束しないことがある（PM7 の HfMe4） | — | 適応混合を 25 回回してから DIIS に切り替え、収束しなければ適応混合 → 定数混合の順に試す（`compute(..., scf_converger=None)` の既定動作）。戻り値の `scf_converged` で確認できる |

## 対応範囲と制限

- PM6・PM7・PM7- とも、PYSEQM がパラメータをもつ元素（H〜Bi のうち MOPAC にパラメータのあるもの）を扱える。照合したのは H, C, N, O, F, Al, Si, Cl, Zr, Hf（下記）。他の遷移金属（Ti, Ta, W など）は同じ経路で動くはずだが MOPAC との照合はしていない。
- d 軌道をもつ系は閉殻（RHF）のみ（PYSEQM の d 経路に UHF が無い）。
- d 経路は sp だけの系より遅い（下の「速度」）。
- MOPAC と同じく、C≡C・NSP2・Si–O–H 補正と分散の炭素 C6 に使う結合リストは最初の構造で固定（`Topology`）。水素結合の供与体・受容体は毎回の構造で判定するので、プロトンが移ると水素結合エネルギーが不連続に変わりうる（MOPAC の PM7 自体の性質）。
- 周期系（MOPAC の id > 0）、COSMO は未対応。
- 物理定数の切り替えはプロセス内の PYSEQM 全体に効く。
- 照合した MOPAC は 23.2.5。23.0.2 では C≡C 補正の式などが違う（PM7 の値も変わる）。

## 照合結果 1: sp 元素のみ（MOPAC 23.2.5、106 系、v0.1）

`validation/` を参照。G2 の H/C/N/O/F 分子 73、S22 二量体 22、水クラスター 8（HF・NH3 + 水 4〜20 分子）、イオン 3。

## 照合結果 2: d 元素を含む系（MOPAC 23.2.5、22 分子 × PM6 / PM7- / PM7、v0.2）

`validation/per_system_d.csv`、`validation/summary_d.txt`、参照値 `validation/mopac_ref_d.json`。構造は GFN2-xTB で最適化したもの（1SCF で比較）。

- 対象: SiH4, AlH3, AlMe3, Al(OH)3, Si(OH)4, H3SiOSiH3, Zr(OH)4, Hf(OH)4, ZrMe4, HfMe4, Zr2O2(OH)4, Al2O3H2、
  ALD 前駆体 TDMAZ（Zr(NMe2)4）, TDMAH（Hf(NMe2)4）, CpZr(NMe2)3, SiH2(NMe2)2, HfCl4, TMA·H2O、
  アダマンタン型クラスター Zr4O6(OH)4, Hf4O6(OH)4, Si4O6(OH)4, Al4O6
- 生成熱: max |ΔHf| = 0.0037（PM6）、0.0019（PM7-）、0.0019（PM7）kcal/mol。SCF は 66/66 で収束
- 力: max |ΔF| = 0.020 kcal/mol/Å（PM7, PM7-）。PM6 の N を含む系（TDMAZ など）で最大 1.6 kcal/mol/Å ずれるが、
  これは MOPAC 側の nsp2 補正の勾配が近似のため。自前の勾配は MOPAC のエネルギーの中心差分と一致する（`scripts/fd_check_d.py`:
  SiH2(NMe2)2 で自前 14.3728・MOPAC のエネルギーの差分 14.3727・MOPAC の勾配 12.7942 kcal/mol/Å）
- 分子の向きに依らない（歪んだ SiCl4 を 4 方向に回して 7.5e-6 kcal/mol 以内。PYSEQM 本体のままでは 0.24）

## 速度（d 経路、PM7、力つき 1 点計算）

| 系 | 原子数 | CPU 4 スレッド | GPU（L4）1 構造 | GPU 8 構造まとめて（1 構造あたり） | MOPAC 23.2.5（1 スレッド） |
|---|---|---|---|---|---|
| TDMAZ | 37 | 3.2 s | 3.5 s | 1.0 s | 0.24 s |
| Zr4O6(OH)4 × 4 | 72 | 13 s | 7.7 s | 2.6 s | 1.05 s |
| TDMAZ × 4 | 148 | 60 s | 6.7 s | 3.4 s | 1.26 s |

（他のプロセスが CPU を使っている共有機での値。torch のスレッド数をコア数いっぱいにすると、共有機では過剰割り当てで桁違いに遅くなる。）
d 経路は MOPAC より 3〜50 倍遅い。時間の大半は PYSEQM の Fock 行列の組み立て（d の一中心・二中心項）と SCF の反復で、
GPU で多数の構造をまとめて計算する使い方が向く。ASE の `SeqmPM7` は既定で前の構造の密度行列を次の SCF の初期値に使う
（Zr(OH)4 の BFGS 27 ステップで 49.5 s → 34.1 s）。

```python
from ase.optimize import BFGS
atoms.calc = SeqmPM7("PM7", device="cuda")        # d 元素を含む系もそのまま
BFGS(atoms).run(fmax=0.02)
r = seqm_pm7.compute(symbols, coords_batch, "PM7", device="cuda")   # (nmol, n, 3) をまとめて
r["scf_converged"], r["scf_converger"]           # SCF の収束と使った方法
```

テスト: `PYTHONPATH=. python -m pytest -q tests/`（62 件、CPU で約 2.5 分）。
