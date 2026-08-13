#!/usr/bin/env python3
"""验证独立清单审计器能识别正确输入与方法配置篡改。

输入：临时源轨迹、资产、历史清单、方法配置和验证清单。
输出：标准unittest通过/失败。
内部逻辑：先验证完整夹具通过，再修改方法配置确认哈希门会拒绝。
作用：保证独立验证的“参数冻结”不是只写在文档里的约定。
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

PREPARE_DIR = Path(__file__).resolve().parents[1] / "prepare"
if str(PREPARE_DIR) not in sys.path:
    sys.path.insert(0, str(PREPARE_DIR))

from create_pilot_manifest import file_sha256
from verify_validation_manifest import verify_validation_manifest


class ValidationManifestAuditTest(unittest.TestCase):
    """覆盖完整通过路径和冻结配置被修改后的拒绝路径。"""

    def build_fixture(self, root):
        """构造一个最小但字段完整的独立验证目录。"""
        source = root / "new.npy"
        np.save(
            source,
            {
                "grasp_seqs": np.zeros((2, 70, 28), dtype=np.float32),
                "obj_rotmat": np.zeros((2, 3, 3), dtype=np.float32),
                "obj_scale": np.ones(2, dtype=np.float32),
            },
            allow_pickle=True,
        )
        asset_root = root / "objects"
        (asset_root / "new").mkdir(parents=True)
        old = root / "old.json"
        old.write_text(json.dumps({"entries": [{"object_name": "old"}]}), encoding="utf-8")
        method = root / "method.json"
        method.write_text(json.dumps({"method": "frozen_v2"}), encoding="utf-8")
        manifest = root / "validation.json"
        manifest.write_text(
            json.dumps(
                {
                    "object_count": 1,
                    "trajectory_count": 2,
                    "excluded_objects": ["old"],
                    "excluded_manifests": [str(old)],
                    "frozen_method": {
                        "path": str(method),
                        "sha256": file_sha256(method),
                        "method": "frozen_v2",
                    },
                    "entries": [
                        {
                            "object_name": "new",
                            "source_path": str(source),
                            "source_sha256": file_sha256(source),
                            "trajectory_indices": [0, 1],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return manifest, method, asset_root

    def test_valid_fixture_passes(self):
        """零泄漏、哈希和资产完整时应返回准确摘要。"""
        with tempfile.TemporaryDirectory() as directory:
            manifest, _, asset_root = self.build_fixture(Path(directory))
            summary = verify_validation_manifest(manifest, asset_root)
            self.assertEqual(summary["object_count"], 1)
            self.assertEqual(summary["trajectory_count"], 2)
            self.assertEqual(summary["frozen_method"], "frozen_v2")

    def test_modified_method_is_rejected(self):
        """清单冻结后修改方法文件必须触发哈希错误。"""
        with tempfile.TemporaryDirectory() as directory:
            manifest, method, asset_root = self.build_fixture(Path(directory))
            method.write_text(json.dumps({"method": "changed"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "哈希已改变"):
                verify_validation_manifest(manifest, asset_root)


if __name__ == "__main__":
    unittest.main()
