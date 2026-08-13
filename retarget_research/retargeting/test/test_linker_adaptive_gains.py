"""测试Linker几何/姿态自适应增益判据。"""

import tempfile
from pathlib import Path
import unittest

import numpy as np

from retarget_research.retargeting.evaluate.evaluate_hand_manifest import (
    linker_adaptive_gain_decision,
)


class LinkerAdaptiveGainsTest(unittest.TestCase):
    """确保规则只在小尺度且闭合角差异明显时触发。"""

    def make_target(self, path, scale, joints):
        """写入包含最小阶段元数据的人工12维候选。"""
        frames = np.zeros((1, 70, 12), dtype=np.float32)
        frames[0, 37, 6:] = joints
        np.save(
            path,
            {
                "grasp_seqs": frames,
                "obj_scale": np.asarray([scale]),
                "squeeze_phase_metadata": [{"lift_start_frame": 37}],
            },
            allow_pickle=True,
        )

    def test_requires_both_small_scale_and_heterogeneous_closure(self):
        """缺少任一条件均保持默认增益。"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "target.npy"
            self.make_target(path, 0.06, [0.5, 0.5, 0.5, 0.5, 1.3, 1.3])
            self.assertTrue(linker_adaptive_gain_decision(path, 0)["use_high_gain"])
            self.make_target(path, 0.08, [0.5, 0.5, 0.5, 0.5, 1.3, 1.3])
            self.assertFalse(linker_adaptive_gain_decision(path, 0)["use_high_gain"])
            self.make_target(path, 0.06, [0.8, 0.8, 0.8, 0.8, 0.8, 0.8])
            self.assertFalse(linker_adaptive_gain_decision(path, 0)["use_high_gain"])


if __name__ == "__main__":
    unittest.main()
