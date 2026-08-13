"""测试候选搜索的物理参数续跑校验和Linker跟踪汇总。

输入：临时摘要与物理报告JSON。
输出：unittest通过或指出错误复用、跟踪统计错误。
内部逻辑：构造最小文件，不启动Isaac Gym。
作用：保证长时间PD搜索不会因resume误用旧结果，也保证解释指标可信。
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


EVALUATE_DIR = Path(__file__).resolve().parents[1] / "evaluate"
sys.path.insert(0, str(EVALUATE_DIR))

from evaluate_candidate_sweep import (  # noqa: E402
    LINKER_ACTIVE_DOFS,
    LINKER_MIMIC_DOFS,
    geometry_step_metrics,
    linker_tracking_metrics,
    load_matching_summary,
)


class CandidateSweepTest(unittest.TestCase):
    """验证搜索结果复用边界和紧凑诊断指标。"""

    def test_resume_rejects_wrong_physics_options(self):
        """输入同轨迹但不同刚度摘要，确认只有准确PD才可续跑。"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.json"
            target = root / "target"
            summary_path = root / "summary.json"
            manifest.write_text("{}", encoding="utf-8")
            target.mkdir()
            summary_path.write_text(
                json.dumps(
                    {
                        "hand": "linker",
                        "manifest": str(manifest),
                        "target_directory": str(target),
                        "trajectory_count": 20,
                        "physics_options": {"linker_finger_stiffness": 120.0},
                    }
                ),
                encoding="utf-8",
            )
            self.assertIsNotNone(
                load_matching_summary(
                    summary_path,
                    "linker",
                    manifest,
                    target,
                    20,
                    {"linker_finger_stiffness": 120.0},
                )
            )
            self.assertIsNone(
                load_matching_summary(
                    summary_path,
                    "linker",
                    manifest,
                    target,
                    20,
                    {"linker_finger_stiffness": 200.0},
                )
            )

    def test_linker_tracking_metrics_groups_active_and_mimic(self):
        """输入已知主动/mimic误差，确认分组平均和峰值正确。"""
        with tempfile.TemporaryDirectory() as temporary:
            report_path = Path(temporary) / "physics.json"
            mean_by_dof = {name: 0.2 for name in LINKER_ACTIVE_DOFS}
            mean_by_dof.update({name: 0.1 for name in LINKER_MIMIC_DOFS})
            max_by_dof = {name: 0.4 for name in mean_by_dof}
            max_by_dof[LINKER_ACTIVE_DOFS[0]] = 0.9
            report_path.write_text(
                json.dumps(
                    {
                        "mean_absolute_tracking_error_by_dof": mean_by_dof,
                        "max_absolute_tracking_error_by_dof": max_by_dof,
                    }
                ),
                encoding="utf-8",
            )
            metrics = linker_tracking_metrics(
                {"results": [{"physics_report": str(report_path)}]}
            )
            self.assertAlmostEqual(metrics["mean_active_tracking_error_rad"], 0.2)
            self.assertAlmostEqual(metrics["mean_mimic_tracking_error_rad"], 0.1)
            self.assertAlmostEqual(metrics["worst_finger_tracking_error_rad"], 0.9)

    def test_geometry_step_metrics_reads_reports(self):
        """输入两份几何报告，确认时序跳变指标按轨迹等权平均。"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reports = []
            for index, scale in enumerate((1.0, 3.0)):
                path = root / f"geometry_{index}.json"
                path.write_text(
                    json.dumps(
                        {
                            "max_joint_step_l2_rad": 0.1 * scale,
                            "max_wrist_translation_step_m": 0.01 * scale,
                            "max_wrist_rotation_step_rad": 0.2 * scale,
                        }
                    ),
                    encoding="utf-8",
                )
                reports.append({"geometry_report": str(path)})
            metrics = geometry_step_metrics({"results": reports})
            self.assertAlmostEqual(metrics["mean_max_joint_step_l2_rad"], 0.2)
            self.assertAlmostEqual(
                metrics["mean_max_wrist_translation_step_m"], 0.02
            )
            self.assertAlmostEqual(metrics["mean_max_wrist_rotation_step_rad"], 0.4)


if __name__ == "__main__":
    unittest.main()
