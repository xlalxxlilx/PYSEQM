"""seqm_pm7: PYSEQM 用の PM7（MOPAC 互換）と、PM6 の MOPAC 互換補正。
電子構造は PYSEQM（sp だけの系は PM6_SP の経路、d 軌道をもつ元素を含む系は spd の経路に PM6/PM7 パラメータを渡す）。
d 経路の PYSEQM の不具合（原子定数表、EISOL、重なり積分、d を含む二中心積分、PM7 の d 処理）は tables / sto_overlap / dint で補う。
補正項は MOPAC 23 のソースを torch に移植したもの。
パラメータと補正式の出典: OpenMOPAC（Apache-2.0, https://github.com/openmopac/mopac）。"""
from .calculator import compute, Topology
from .ase_calc import SeqmPM7
__version__ = "0.2.0"
