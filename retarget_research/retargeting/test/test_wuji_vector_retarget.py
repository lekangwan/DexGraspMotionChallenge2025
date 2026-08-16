"""测试Wuji功能向量配置、Huber损失和安全续跑识别。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

import numpy as np
import torch


RETARGET_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RETARGET_ROOT / "run"))

from retarget_wuji_vectors import (  # noqa: E402
    DEFAULT_VECTOR_CONFIG,
    huber_norm,
    load_vector_config,
)
from run_wuji_vector_manifest import existing_output_matches  # noqa: E402
sys.path.insert(0, str(RETARGET_ROOT / "evaluate"))
from evaluate_hand_manifest import geometry_script_for_target  # noqa: E402
from analyze_wuji_thumb_nullspace import (  # noqa: E402
    angle_statistics,
    displacement_statistics,
)


class WujiVectorRetargetTest(unittest.TestCase):
    """锁定12向量定义、Huber数值和向量候选不能与旧基线混用。"""

    def test_vector_config_uses_only_palm_and_fingertip_relations(self):
        """配置应有12条唯一语义，并覆盖5条掌指和4条拇指对指向量。"""
        config, path, digest = load_vector_config(DEFAULT_VECTOR_CONFIG)
        semantics = {item["semantic"] for item in config["pairs"]}
        self.assertEqual(len(semantics), 12)
        self.assertEqual(sum(name.startswith("palm_to_") for name in semantics), 5)
        self.assertEqual(sum(name.startswith("thumb_to_") for name in semantics), 4)
        self.assertEqual(Path(path), DEFAULT_VECTOR_CONFIG.resolve())
        self.assertEqual(len(digest), 64)

    def test_huber_norm_is_quadratic_then_linear(self):
        """1 cm残差应处于二次段，3 cm残差应进入2 cm阈值后的线性段。"""
        residual = torch.tensor([[0.01, 0.0, 0.0], [0.03, 0.0, 0.0]])
        actual = huber_norm(residual, 0.02).numpy()
        np.testing.assert_allclose(actual, [0.00005, 0.0004], rtol=1e-5)

    def test_resume_requires_vector_method_hash(self):
        """即使形状和索引相同，缺少向量方法元数据的旧npy也必须拒绝复用。"""
        entry = {"trajectory_indices": [4]}
        args = SimpleNamespace(
            vector_config=DEFAULT_VECTOR_CONFIG.resolve(),
            anatomy_config=None,
            maxeval=50,
            source_z_offset=0.4,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "old.npy"
            np.save(
                output,
                {
                    "grasp_seqs": np.zeros((1, 70, 26), dtype=np.float32),
                    "source_trajectory_indices": np.asarray([4]),
                    "maxeval": 50,
                    "source_z_offset": 0.4,
                },
                allow_pickle=True,
            )
            self.assertFalse(existing_output_matches(output, entry, args))

    def test_manifest_selects_vector_geometry_without_affecting_old_wuji(self):
        """纯向量/接触混合使用向量评估，旧Wuji继续使用15点评估。"""
        with tempfile.TemporaryDirectory() as directory:
            vector_path = Path(directory) / "vector.npy"
            hybrid_path = Path(directory) / "hybrid.npy"
            legacy_path = Path(directory) / "legacy.npy"
            np.save(vector_path, {"retarget_method": "dexpilot_style_functional_vectors_v1"}, allow_pickle=True)
            np.save(
                hybrid_path,
                {
                    "retarget_method": (
                        "dexpilot_style_functional_vectors_plus_surface_contact_v1"
                    )
                },
                allow_pickle=True,
            )
            np.save(legacy_path, {"mapping_semantics": ["palm"]}, allow_pickle=True)
            self.assertEqual(
                geometry_script_for_target("wuji", vector_path).name,
                "evaluate_wuji_vector_geometry.py",
            )
            self.assertEqual(
                geometry_script_for_target("wuji", hybrid_path).name,
                "evaluate_wuji_vector_geometry.py",
            )
            self.assertEqual(
                geometry_script_for_target("wuji", legacy_path).name,
                "evaluate_wuji_geometry.py",
            )

    def test_thumb_angle_statistics_count_near_ninety(self):
        """拇指审计应正确统计85–95度区间和92度上界区。"""
        report = angle_statistics([40.0, 85.0, 90.0, 92.0, 96.0])
        self.assertAlmostEqual(report["near_85_to_95_ratio"], 3 / 5)
        self.assertAlmostEqual(report["at_or_above_92_ratio"], 2 / 5)
        self.assertEqual(report["median_deg"], 90.0)

    def test_thumb_displacement_statistics_convert_to_millimeters(self):
        """指尖偏移输入为米，报告必须统一转为毫米。"""
        report = displacement_statistics([0.001, 0.002, 0.003])
        self.assertAlmostEqual(report["mean_mm"], 2.0)
        self.assertAlmostEqual(report["maximum_mm"], 3.0)


if __name__ == "__main__":
    unittest.main()
