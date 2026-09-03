#!/usr/bin/env python3
"""成功率补救工具的快速纯逻辑测试，不启动Isaac Gym。"""

from compare_formal_results import compare
from prepare_failure_manifest import failure_keys, subset_manifest
from run_confirmed_synergy_manifest import robust_decision


def metric(success, transport, final_lift=0.2):
    """构造physics_score所需的最小测试指标。"""
    return {
        "success": success,
        "transport_stability_success": transport,
        "final_lift_m": final_lift,
        "max_lift_m": final_lift,
        "terminal_contact_ratio": float(success),
        "hand_object_contact_steps": 100,
        "peak_to_final_drop_m": 0.0,
        "max_xy_drift_m": 0.0,
        "max_palm_relative_translation_change_m": 0.0 if transport else 0.05,
        "max_palm_relative_rotation_change_deg": 0.0 if transport else 40.0,
    }


def test_failure_subset():
    """失败键必须准确映射回原manifest中的轨迹索引。"""
    audit = {"results": [
        {"object_name": "a", "source_trajectory_index": 1,
         "transport_quality_success": True},
        {"object_name": "a", "source_trajectory_index": 2,
         "transport_quality_success": False},
    ]}
    manifest = {"purpose": "all", "entries": [{
        "object_name": "a", "trajectory_indices": [1, 2],
        "calibration_indices": [1], "heldout_indices": [2],
    }]}
    failures = failure_keys(audit, "transport")
    result = subset_manifest(manifest, failures, "transport", __import__("pathlib").Path("x"))
    assert result["trajectory_count"] == 1
    assert result["entries"][0]["trajectory_indices"] == [2]
    limited = subset_manifest(
        manifest, failures, "transport", __import__("pathlib").Path("x"), 1
    )
    assert limited["trajectory_count"] == 1


def test_comparison_gate():
    """运输净增不足5个百分点时必须开启补救分支。"""
    old = {"trajectory_count": 100, "source_10cm_success_count": 60,
           "stable_physics_success_count": 58,
           "transport_quality_success_count": 55}
    new = {"trajectory_count": 100, "source_10cm_success_count": 63,
           "stable_physics_success_count": 61,
           "transport_quality_success_count": 59}
    assert compare(old, new, 0.05)["decision"] == "open_success_only_recovery"


def test_robust_confirmation():
    """两次都成功才接受；任意一次运输失败必须拒绝。"""
    failed = [metric(False, False), metric(False, False)]
    passed = [metric(True, True), metric(True, True)]
    unstable = [metric(True, True), metric(True, False)]
    assert robust_decision(failed, passed, 1.0)["accepted"]
    assert not robust_decision(failed, unstable, 1.0)["accepted"]


if __name__ == "__main__":
    test_failure_subset()
    test_comparison_gate()
    test_robust_confirmation()
    print("3 success-only tests passed")
