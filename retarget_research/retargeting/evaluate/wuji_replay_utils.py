"""提供Wuji优化输出到Isaac Gym DOF顺序的纯数据映射。

输入：保存的26维帧、20个优化关节名和Isaac实际DOF名称。
输出：按Isaac名称顺序排列的26维位置目标。
内部逻辑：分别给手腕6维和手指20维建立名称到数值字典，再按物理顺序查询。
作用：防止URDF遍历顺序变化后动作仍能执行，却悄悄控制了错误关节。
"""

import numpy as np


WRIST_NAMES = [
    "virtual_joint_x",
    "virtual_joint_y",
    "virtual_joint_z",
    "virtual_joint_roll",
    "virtual_joint_pitch",
    "virtual_joint_yaw",
]


def reorder_wuji_frame(frame, optimizer_joint_names, physics_dof_names):
    """按DOF名称重排一帧Wuji候选动作。

    输入：`[手腕6,手指20]`、保存的手指名和Isaac返回的26个DOF名。
    输出：与`physics_dof_names`同顺序的float32数组。
    内部逻辑：校验维度与名称集合，建立26项字典后逐名读取。
    作用：让数学模型与物理引擎只通过关节名称连接，不依赖隐式索引。
    """
    frame = np.asarray(frame, dtype=np.float32)
    optimizer_joint_names = list(optimizer_joint_names)
    physics_dof_names = list(physics_dof_names)
    if frame.shape != (26,) or len(optimizer_joint_names) != 20:
        raise ValueError("Wuji帧必须为26维且包含20个优化关节名")
    value_by_name = {
        name: float(frame[index]) for index, name in enumerate(WRIST_NAMES)
    }
    value_by_name.update(
        {
            name: float(frame[index + 6])
            for index, name in enumerate(optimizer_joint_names)
        }
    )
    if set(value_by_name) != set(physics_dof_names):
        missing = sorted(set(physics_dof_names) - set(value_by_name))
        extra = sorted(set(value_by_name) - set(physics_dof_names))
        raise ValueError(f"Wuji DOF名称不一致，缺少{missing}，多出{extra}")
    return np.asarray(
        [value_by_name[name] for name in physics_dof_names], dtype=np.float32
    )
