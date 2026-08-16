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


if __name__ == "__main__":
    unittest.main()
