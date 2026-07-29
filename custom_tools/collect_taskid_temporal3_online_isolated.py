"""Collect Temporal3-visited states on the same train trajectories as old R2."""

from pathlib import Path
import sys

from custom_tools import collect_taskid_online_scaled20_isolated as collector


ROOT = Path(__file__).resolve().parents[1]
collector.BC_CONFIG = (
    ROOT / "custom_tools/configs/"
    / "unified_student_taskid_temporal3_v1.yaml"
)
collector.STUDENT = (
    ROOT / "custom_tools/runs/bc/"
    / "unified_student_taskid_temporal3_seed2025_e4_v1/"
    / "epoch=003-step=5152.ckpt"
)
collector.OUTPUT = (
    ROOT / "custom_tools/data/distillation/"
    / "online_taskid_temporal3_r1_train4_offset4.npz"
)
collector.PARTS = (
    ROOT / "custom_tools/data/distillation/"
    / "online_taskid_temporal3_r1_train4_offset4_parts"
)


def main():
    # Match old online R2 exactly: same staged trajectory positions 4--7.
    # The only intended difference is the rollout policy (R1 vs Temporal3).
    if "--trajectory-start-offset" not in sys.argv:
        sys.argv.extend(["--trajectory-start-offset", "4"])
    collector.main()


if __name__ == "__main__":
    main()
