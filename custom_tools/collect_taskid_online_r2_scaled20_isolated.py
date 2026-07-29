"""Collect Task-ID online-imitation round 2 on fresh training trajectories."""

from pathlib import Path
import sys

from custom_tools import collect_taskid_online_scaled20_isolated as collector


ROOT = Path(__file__).resolve().parents[1]
collector.BC_CONFIG = (
    ROOT / "custom_tools/configs/"
    / "unified_student_taskid_online_r1_scaled20_v1.yaml"
)
collector.STUDENT = (
    ROOT / "custom_tools/runs/bc/"
    / "unified_student_taskid_online_r1_frac025_seed2025_e10_v1/"
    / "epoch=001-step=2232.ckpt"
)
collector.OUTPUT = (
    ROOT / "custom_tools/data/distillation/"
    / "online_taskid_scaled20_r2_train4_offset4.npz"
)
collector.PARTS = (
    ROOT / "custom_tools/data/distillation/"
    / "online_taskid_scaled20_r2_train4_offset4_parts"
)


def main():
    if "--trajectory-start-offset" not in sys.argv:
        sys.argv.extend(["--trajectory-start-offset", "4"])
    collector.main()


if __name__ == "__main__":
    main()
