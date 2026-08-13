#!/usr/bin/env python3
"""验证独立manifest的防泄漏和方法冻结元数据。

输入：临时历史manifest、临时方法配置和少量显式排除名称。
输出：标准unittest通过/失败。
内部逻辑：检查历史条目、历史排除项和显式名称的并集，并核对配置哈希。
作用：防止新验证集意外重复使用开发物体或无法证明所用参数版本。
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from retarget_research.retargeting.prepare.create_pilot_manifest import (
    collect_excluded_objects,
    file_sha256,
    frozen_method_record,
)


class IndependentManifestTest(unittest.TestCase):
    """覆盖历史物体排除与冻结配置记录两个纯文件规则。"""

    def test_collect_excluded_objects_merges_all_sources(self):
        """历史entries、历史排除字段和显式名称都必须进入排除集合。"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.json"
            path.write_text(
                json.dumps(
                    {
                        "entries": [{"object_name": "evaluated"}],
                        "excluded_tuning_objects": ["tuned"],
                        "excluded_objects": ["older"],
                    }
                ),
                encoding="utf-8",
            )
            excluded, manifests = collect_excluded_objects(["explicit"], [path])
            self.assertEqual(excluded, {"evaluated", "tuned", "older", "explicit"})
            self.assertEqual(manifests, [str(path.resolve())])

    def test_frozen_method_record_has_exact_hash(self):
        """冻结记录必须包含配置方法名、绝对路径和原文件精确哈希。"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "method.json"
            path.write_text(json.dumps({"method": "fixed_v2"}), encoding="utf-8")
            record = frozen_method_record(path)
            self.assertEqual(record["method"], "fixed_v2")
            self.assertEqual(record["path"], str(path.resolve()))
            self.assertEqual(record["sha256"], file_sha256(path))


if __name__ == "__main__":
    unittest.main()
