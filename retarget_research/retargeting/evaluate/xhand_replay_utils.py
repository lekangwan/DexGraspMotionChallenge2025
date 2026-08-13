"""提供XHand优化输出到Isaac物理DOF的纯名称映射。

输入：18维`[手腕6,手指12]`候选帧和Isaac DOF名称顺序。
输出：按物理actor顺序重排的18维目标。
内部逻辑：先按优化器固定语义顺序建名称字典，再按Isaac名称查询。
作用：解决运动学模型和Isaac Gym手指关节遍历顺序不同的问题。
"""

import numpy as np


OPTIMIZER_FINGER_NAMES = [
    "right_hand_thumb_bend_joint",
    "right_hand_thumb_rota_joint1",
    "right_hand_thumb_rota_joint2",
    "right_hand_index_bend_joint",
    "right_hand_index_joint1",
    "right_hand_index_joint2",
    "right_hand_mid_joint1",
    "right_hand_mid_joint2",
    "right_hand_ring_joint1",
    "right_hand_ring_joint2",
    "right_hand_pinky_joint1",
    "right_hand_pinky_joint2",
]

WRIST_NAMES = [
    "virtual_x",
    "virtual_y",
    "virtual_z",
    "virtual_rx",
    "virtual_ry",
    "virtual_rz",
]


def reorder_xhand_frame(frame, dof_names):
    """按名称把XHand 18维保存帧重排为Isaac DOF顺序。

    输入：优化器顺序的18维帧和Isaac返回的18个DOF名称。
    输出：与物理actor顺序一致的浮点数组。
    逻辑：手腕前6维直接绑定虚拟关节，后12维绑定运动学模型名称。
    作用：避免Isaac把食指开合量误发送给拇指或其他手指。
    """
    frame = np.asarray(frame, dtype=np.float32)
    if frame.shape != (18,):
        raise ValueError(f"单帧XHand候选应为18维，实际为{frame.shape}")
    values = dict(zip(WRIST_NAMES, frame[:6]))
    values.update(dict(zip(OPTIMIZER_FINGER_NAMES, frame[6:])))
    unknown = [name for name in dof_names if name not in values]
    if unknown:
        raise ValueError(f"未知XHand物理DOF名称: {unknown}")
    return np.asarray([values[name] for name in dof_names], dtype=np.float32)
