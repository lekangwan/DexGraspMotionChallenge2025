"""测试终态稳定成功与掌物相对滑移的新硬门。"""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


RETARGET_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RETARGET_ROOT / "evaluate"))

from audit_stable_success import terminal_metrics, transport_metrics


class StableSuccessTest(unittest.TestCase):
    """验证新成功定义不会再放过瞬时抬起或掌内滑移。"""

    def test_lift_then_drop_keeps_legacy_but_fails_stable_success(self):
        """前30步越线、末30步掉回桌面必须被新口径拒绝。"""
        positions = np.zeros((60, 3), dtype=np.float64)
        positions[:30, 2] = 0.12
        contacts = np.ones(60, dtype=np.int64)
        report = {
            "object_positions_m": positions.tolist(),
            "initial_object_position_m": [0.0, 0.0, 0.0],
            "hand_object_contact_count_per_step": contacts.tolist(),
            "max_allowed_xy_drift_m": 0.25,
            "required_sustain_steps": 30,
        }
        protocol = {
            "terminal_hold_steps": 30,
            "lift_threshold_m": 0.10,
            "max_peak_to_final_drop_m": 0.03,
            "max_terminal_lift_range_m": 0.01,
            "min_terminal_contact_ratio": 1.0,
        }
        metrics = terminal_metrics(report, protocol)
        self.assertTrue(metrics["legacy_success_recomputed"])
        self.assertFalse(metrics["stable_physics_success"])

    def test_complete_terminal_hold_is_stable_success(self):
        """全程高于阈值、持续接触且无波动的轨迹应通过。"""
        positions = np.zeros((60, 3), dtype=np.float64)
        positions[:, 2] = 0.12
        report = {
            "object_positions_m": positions.tolist(),
            "initial_object_position_m": [0.0, 0.0, 0.0],
            "hand_object_contact_count_per_step": np.ones(60, dtype=int).tolist(),
            "max_allowed_xy_drift_m": 0.25,
            "required_sustain_steps": 30,
        }
        protocol = {
            "terminal_hold_steps": 30,
            "lift_threshold_m": 0.10,
            "max_peak_to_final_drop_m": 0.03,
            "max_terminal_lift_range_m": 0.01,
            "min_terminal_contact_ratio": 1.0,
        }
        metrics = terminal_metrics(report, protocol)
        self.assertTrue(metrics["legacy_success_recomputed"])
        self.assertTrue(metrics["stable_physics_success"])

    def test_transport_metric_distinguishes_carried_from_sliding_object(self):
        """同样抬升的物体，只有相对手掌位置不变才能通过运输门。"""
        protocol = {
            "start_after_lift_m": 0.05,
            "max_palm_relative_translation_change_m": 0.03,
            "max_palm_relative_rotation_change_deg": 30.0,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            steps = 60
            wrist = np.zeros((steps, 6), dtype=np.float32)
            wrist[:, 2] = np.linspace(0.0, 0.12, steps)
            quaternion = np.tile(
                np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                (steps, 1),
            )
            contact = np.ones(steps, dtype=np.int64)
            carried = root / "carried.npz"
            np.savez_compressed(
                carried,
                hand_dof_position=wrist,
                object_position=wrist[:, :3] + np.asarray([0.0, 0.0, 0.06]),
                object_quaternion_xyzw=quaternion,
                hand_object_contact_count=contact,
            )
            carried_metrics = transport_metrics(carried, protocol, 20)
            self.assertTrue(carried_metrics["transport_stability_success"])

            sliding = root / "sliding.npz"
            sliding_object = wrist[:, :3] + np.asarray([0.0, 0.0, 0.06])
            sliding_object[:, 0] += np.linspace(0.0, 0.08, steps)
            np.savez_compressed(
                sliding,
                hand_dof_position=wrist,
                object_position=sliding_object,
                object_quaternion_xyzw=quaternion,
                hand_object_contact_count=contact,
            )
            sliding_metrics = transport_metrics(sliding, protocol, 20)
            self.assertFalse(sliding_metrics["transport_stability_success"])


if __name__ == "__main__":
    unittest.main()
