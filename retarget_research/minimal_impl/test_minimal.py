"""最小实现的CPU单元测试。

输入：随机张量和人工物体位置曲线。
输出：``MINIMAL_RETARGET_TEST=PASS``。
逻辑：检查三手运动学维度、动作映射、稳定运输判据和Rank-5轨迹变换。
作用：不运行长时间PhysX，也能发现最常见的接口和数学错误。
"""

import numpy as np
import torch

from retarget_research.minimal_impl.config import HANDS, WRIST_LINKER_WUJI
from retarget_research.minimal_impl.kinematics import (
    build_shadow_model, build_target_model, joint_names, shadow_keypoints, target_keypoints,
)
from retarget_research.minimal_impl.cem import (
    apply_global, apply_synergy, build_synergy_basis, robust_confirmation,
)
from retarget_research.minimal_impl.simulate import map_command, success_metrics


def test_kinematics():
    """输入零姿态，输出三手关键点形状断言；作用是检查资产路径和主动关节维度。"""
    shadow = build_shadow_model()
    assert shadow_keypoints(np.zeros((2, 28), dtype=np.float32), shadow).shape == (2, 21, 3)
    for hand, spec in HANDS.items():
        model = build_target_model(hand)
        assert int(model.revolute_joints_q_lower.shape[1]) == spec.finger_dim
        points = target_keypoints(
            model, hand, torch.zeros(spec.finger_dim), torch.zeros(3), torch.zeros(3),
        )
        assert points.ndim == 2 and points.shape[1] == 3


def test_mapping_and_success():
    """输入人工动作/位置曲线，输出映射和成功断言；作用是检查mimic倍率及末段稳定规则。"""
    linker_names = list(WRIST_LINKER_WUJI) + [
        "rh_thumb_cmc_yaw", "rh_thumb_cmc_pitch", "rh_thumb_ip",
        "rh_index_mcp_pitch", "rh_index_dip", "rh_middle_mcp_pitch", "rh_middle_dip",
        "rh_ring_mcp_pitch", "rh_ring_dip", "rh_pinky_mcp_pitch", "rh_pinky_dip",
    ]
    physical = map_command("linker", np.arange(12, dtype=np.float32), linker_names, {})
    assert physical.shape == (17,)
    assert np.isclose(physical[8], 7 * 1.86)

    positions = np.zeros((60, 3), dtype=np.float32)
    positions[:, 2] = np.linspace(0.0, 0.32, 60)
    positions[-30:, 2] = 0.32
    contacts = np.ones(60, dtype=np.int64)
    hand_poses = np.zeros((60, 6), dtype=np.float32)
    hand_poses[:, 2] = positions[:, 2]
    quaternions = np.tile([0, 0, 0, 1], (60, 1)).astype(np.float32)
    metric = success_metrics(
        positions, np.zeros(3), contacts, hand_poses, quaternions,
    )
    assert metric["reference_isaac_success"]
    assert metric["success"] and metric["transport_stability_success"]
    positions[-1, 2] = 0.0
    assert not success_metrics(positions, np.zeros(3), contacts)["success"]


def test_cem_components():
    """输入人工70帧轨迹，输出Global/Rank-5形状与重复确认断言；作用是检查最终方法核心数学。"""
    trajectories = np.zeros((6, 70, 18), dtype=np.float32)
    time = np.linspace(0.0, 1.0, 70, dtype=np.float32)
    for index in range(6):
        trajectories[index, :, 6:] = time[:, None] * (index + 1) * 0.02
        trajectories[index, :, 2] = np.maximum(time - 0.6, 0.0)
    basis = build_synergy_basis(trajectories, rank=5)
    assert basis.shape[1] == 12 and 1 <= len(basis) <= 5
    assert apply_global(trajectories[0], np.zeros(12)).shape == (70, 18)
    assert apply_synergy(
        trajectories[0], np.zeros(2 * len(basis)), basis,
    ).shape == (70, 18)

    def evaluate(values):
        """输入人工候选参数，输出可控的成功/失败指标，用于测试确认门。"""
        passing = bool(np.any(values[0] != 0.0))
        return [{
            "success": passing, "transport_stability_success": passing,
            "final_lift_m": 0.2 if passing else 0.0,
            "max_lift_m": 0.2 if passing else 0.0,
            "terminal_contact_ratio": float(passing),
            "hand_object_contact_steps": 100 if passing else 0,
            "peak_to_final_drop_m": 0.0,
            "max_xy_drift_m": 0.0,
            "max_palm_relative_translation_change_m": 0.0,
            "max_palm_relative_rotation_change_deg": 0.0,
        }]

    accepted, _, _ = robust_confirmation(evaluate, np.ones(4, dtype=np.float32))
    assert accepted


def main():
    """依次运行三个测试组；输入无，输出PASS标记。"""
    torch.manual_seed(7)
    test_kinematics()
    test_mapping_and_success()
    test_cem_components()
    print("MINIMAL_RETARGET_TEST=PASS")


if __name__ == "__main__":
    main()
