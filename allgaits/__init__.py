"""AllGaits replication on Unitree B1.

Canonical leg ordering used throughout this package: [FL, FR, RL, RR].
This differs from the paper's matrix ordering [FR, FL, HR, HL] — the Φ
matrices in `allgaits.cpg.coupling` are expressed in FL/FR/RL/RR order.
"""

LEG_ORDER = ("FL", "FR", "RL", "RR")
NUM_LEGS = 4
