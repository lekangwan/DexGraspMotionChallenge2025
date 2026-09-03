import unittest

import numpy as np

from retarget_research.retargeting.run.physics_cem_refine import (
    apply_trajectory_parameters, physics_score,
)
from retarget_research.advanced_policy.train_residual_ppo_general import (
    interpolated_policy_command,
)


class PhysicsCEMTest(unittest.TestCase):
    def test_zero_parameters_preserve_trajectory(self):
        frames = np.random.default_rng(0).normal(size=(70, 18)).astype(np.float32)
        np.testing.assert_allclose(
            apply_trajectory_parameters(frames, np.zeros(12)), frames)

    def test_late_finger_offset_preserves_first_frame(self):
        frames = np.zeros((70, 18), dtype=np.float32)
        parameters = np.zeros(12, dtype=np.float32)
        parameters[6:11] = 0.1
        changed = apply_trajectory_parameters(frames, parameters)
        np.testing.assert_allclose(changed[0], frames[0])
        self.assertGreater(float(np.abs(changed[-1, 6:]).sum()), 0.0)

    def test_linker_six_active_joints_have_independent_offsets(self):
        frames = np.zeros((70, 12), dtype=np.float32)
        parameters = np.zeros(13, dtype=np.float32)
        parameters[6:12] = np.arange(1, 7, dtype=np.float32) * 0.01
        changed = apply_trajectory_parameters(frames, parameters)
        np.testing.assert_allclose(changed[-1, 6:], parameters[6:12])

    def test_stable_success_dominates_lift_only(self):
        common = {
            "final_lift_m": 0.30, "max_lift_m": 0.31,
            "terminal_contact_ratio": 1.0,
            "hand_object_contact_steps": 150,
            "peak_to_final_drop_m": 0.01, "max_xy_drift_m": 0.01,
        }
        self.assertGreater(
            physics_score({**common, "success": True}),
            physics_score({**common, "success": False}))

    def test_transport_quality_breaks_stable_success_tie(self):
        common = {
            "success": True, "final_lift_m": 0.30, "max_lift_m": 0.31,
            "terminal_contact_ratio": 1.0, "hand_object_contact_steps": 150,
            "peak_to_final_drop_m": 0.01, "max_xy_drift_m": 0.01,
            "max_palm_relative_translation_change_m": 0.01,
            "max_palm_relative_rotation_change_deg": 5.0,
        }
        self.assertGreater(
            physics_score({**common, "transport_stability_success": True}),
            physics_score({**common, "transport_stability_success": False}))

    def test_physics_steps_match_standard_three_step_interpolation(self):
        frames = np.asarray([[3.0], [6.0]], dtype=np.float32)
        opened = np.asarray([0.0], dtype=np.float32)
        values = [
            interpolated_policy_command(frames, opened, step, 3)[0]
            for step in range(7)
        ]
        np.testing.assert_allclose(values, [1, 2, 3, 4, 5, 6, 6])


if __name__ == "__main__":
    unittest.main()
