"""提供Wuji优化输出到Isaac Gym DOF顺序及解剖限位的纯数据适配。

输入：保存的26维帧、20个优化关节名和Isaac实际DOF名称。
输出：按Isaac名称顺序排列的26维位置目标。
内部逻辑：分别给手腕6维和手指20维建立名称到数值字典，再按物理顺序查询。
作用：防止URDF遍历顺序变化后动作仍能执行，却悄悄控制了错误关节。
"""

import hashlib
import json
from pathlib import Path

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


def apply_anatomy_dof_limits(properties, physics_dof_names, target_data):
    """把候选使用的解剖边界同步写入Isaac物理DOF属性。

    输入：Isaac结构化DOF属性、物理关节名和候选npy字典。
    输出：是否启用、配置路径/哈希和实际覆盖值组成的元数据字典。
    内部逻辑：候选无手型配置时保持旧基线；有配置时先核对文件SHA-256，
    再只允许收紧现有上下界，并把对应物理轴标为有限位。
    作用：避免优化命令已自然、真实关节却在接触力下仍被顶回反向极限。
    """
    config_value = target_data.get("anatomy_config")
    if config_value is None:
        return {
            "anatomy_limits_enforced": False,
            "anatomy_config": None,
            "anatomy_config_sha256": None,
            "anatomy_lower_bounds_rad": {},
            "anatomy_upper_bounds_rad": {},
        }
    config_path = Path(str(config_value)).resolve()
    raw = config_path.read_bytes()
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    expected_sha256 = str(target_data.get("anatomy_config_sha256", ""))
    if actual_sha256 != expected_sha256:
        raise ValueError("Wuji候选的手型配置SHA-256与当前文件不一致")
    config = json.loads(raw.decode("utf-8"))
    by_name = {name: index for index, name in enumerate(physics_dof_names)}
    applied_lower, applied_upper = {}, {}
    for field, target, applied, is_lower in (
        ("lower_bound_overrides_rad", "lower", applied_lower, True),
        ("upper_bound_overrides_rad", "upper", applied_upper, False),
    ):
        for name, raw_value in config.get(field, {}).items():
            if name not in by_name:
                raise ValueError(f"物理解剖配置含未知DOF: {name}")
            index = by_name[name]
            value = float(raw_value)
            current_lower = float(properties["lower"][index])
            current_upper = float(properties["upper"][index])
            valid = (
                np.isfinite(value)
                and (current_lower <= value < current_upper if is_lower else current_lower < value <= current_upper)
            )
            if not valid:
                raise ValueError(f"{name}的物理解剖边界不能放宽或反转URDF范围")
            properties[target][index] = value
            if "hasLimits" in properties.dtype.names:
                properties["hasLimits"][index] = True
            applied[name] = value
    return {
        "anatomy_limits_enforced": True,
        "anatomy_config": str(config_path),
        "anatomy_config_sha256": actual_sha256,
        "anatomy_lower_bounds_rad": applied_lower,
        "anatomy_upper_bounds_rad": applied_upper,
    }
