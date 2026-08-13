"""提供不依赖Isaac Gym的Linker物理重放纯数据函数。

输入：12维O6候选帧、17维解耦候选帧、目标DOF名称或一维布尔序列。
输出：按名称展开的17维目标或最长连续真值长度。
内部逻辑：显式应用Linker mimic倍率，并用单次扫描统计连续区间。
作用：隔离可单元测试的数据规则，避免测试被Isaac/PyTorch导入顺序绑住。
"""

import numpy as np


MIMIC_DOF_NAMES = {
    "rh_thumb_ip",
    "rh_index_dip",
    "rh_middle_dip",
    "rh_ring_dip",
    "rh_pinky_dip",
}


def linker_dof_gains(
    dof_names,
    finger_stiffness=120.0,
    finger_damping=5.0,
    mimic_stiffness=120.0,
    mimic_damping=5.0,
):
    """按手腕、主动手指和从动手指生成位置控制增益。

    输入：Isaac DOF名称，以及主动/从动手指的刚度和阻尼。
    输出：与名称顺序一致的`(stiffness, damping)`两个浮点数组。
    内部逻辑：手腕继续使用固定高增益；5个IP/DIP从动轴使用mimic参数；
    其余6个手指主动轴使用finger参数。
    作用：默认120/5完全复现旧基线；降低mimic刚度时仍保持6维命令，
    但允许从动指节在接触作用下偏离固定倍率目标。
    """
    values = [finger_stiffness, finger_damping, mimic_stiffness, mimic_damping]
    if any(float(value) < 0 for value in values):
        raise ValueError("DOF刚度和阻尼必须大于等于0")
    stiffness, damping = [], []
    for name in dof_names:
        if name in {"virtual_joint_x", "virtual_joint_y", "virtual_joint_z"}:
            stiffness.append(20000.0)
            damping.append(500.0)
        elif name in {
            "virtual_joint_roll",
            "virtual_joint_pitch",
            "virtual_joint_yaw",
        }:
            stiffness.append(2000.0)
            damping.append(80.0)
        elif name in MIMIC_DOF_NAMES:
            stiffness.append(float(mimic_stiffness))
            damping.append(float(mimic_damping))
        else:
            stiffness.append(float(finger_stiffness))
            damping.append(float(finger_damping))
    return np.asarray(stiffness, dtype=np.float32), np.asarray(damping, dtype=np.float32)


def expand_active_frame(frame, dof_names):
    """把12维候选帧展开并重排为Isaac的17维DOF目标。

    输入：`[手腕6,主动关节6]`帧和Isaac实际DOF名称顺序。
    输出：与物理actor DOF顺序一致的浮点数组。
    逻辑：按URDF倍率生成5个mimic关节，再以名称而非数字索引重排。
    作用：保证仿真器不支持mimic标签时，目标手仍执行正确机械联动。
    """
    frame = np.asarray(frame, dtype=np.float32)
    if frame.shape != (12,):
        raise ValueError(f"单帧Linker候选应为12维，实际为{frame.shape}")
    active = frame[6:]
    values = {
        "virtual_joint_x": frame[0],
        "virtual_joint_y": frame[1],
        "virtual_joint_z": frame[2],
        "virtual_joint_roll": frame[3],
        "virtual_joint_pitch": frame[4],
        "virtual_joint_yaw": frame[5],
        "rh_thumb_cmc_yaw": active[0],
        "rh_thumb_cmc_pitch": active[1],
        "rh_thumb_ip": active[1] * 1.86,
        "rh_index_mcp_pitch": active[2],
        "rh_index_dip": active[2] * 0.89,
        "rh_middle_mcp_pitch": active[3],
        "rh_middle_dip": active[3] * 0.89,
        "rh_ring_mcp_pitch": active[4],
        "rh_ring_dip": active[4] * 0.89,
        "rh_pinky_mcp_pitch": active[5],
        "rh_pinky_dip": active[5] * 0.89,
    }
    unknown = [name for name in dof_names if name not in values]
    if unknown:
        raise ValueError(f"未知Linker物理DOF名称: {unknown}")
    return np.asarray([values[name] for name in dof_names], dtype=np.float32)


def expand_independent_frame(frame, dof_names):
    """把17维解耦候选帧按名称重排成Isaac的17维目标。

    输入：`[手腕6,完整手指关节11]`以及Isaac实际DOF名称顺序。
    输出：与物理actor DOF顺序一致的浮点数组。
    逻辑：候选中的11个角度逐一对应URDF关节，不再使用1.86/0.89倍率。
    作用：让同一Linker外形在仿真中真正执行优化器新增的5个独立控制量。
    """
    frame = np.asarray(frame, dtype=np.float32)
    if frame.shape != (17,):
        raise ValueError(f"单帧Linker解耦候选应为17维，实际为{frame.shape}")
    ordered_names = [
        "virtual_joint_x",
        "virtual_joint_y",
        "virtual_joint_z",
        "virtual_joint_roll",
        "virtual_joint_pitch",
        "virtual_joint_yaw",
        "rh_thumb_cmc_yaw",
        "rh_thumb_cmc_pitch",
        "rh_thumb_ip",
        "rh_index_mcp_pitch",
        "rh_index_dip",
        "rh_middle_mcp_pitch",
        "rh_middle_dip",
        "rh_ring_mcp_pitch",
        "rh_ring_dip",
        "rh_pinky_mcp_pitch",
        "rh_pinky_dip",
    ]
    values = dict(zip(ordered_names, frame))
    unknown = [name for name in dof_names if name not in values]
    if unknown:
        raise ValueError(f"未知Linker物理DOF名称: {unknown}")
    return np.asarray([values[name] for name in dof_names], dtype=np.float32)


def expand_linker_frame(frame, dof_names):
    """根据候选维度选择真实O6展开或11轴直接控制。

    输入：一帧12/17维Linker候选及Isaac DOF名称。
    输出：统一17维物理位置目标。
    逻辑：12维调用mimic展开，17维调用逐关节重排，其他维度立即报错。
    作用：使物理入口兼容两种实验，同时杜绝静默套用错误的机械关系。
    """
    dimension = np.asarray(frame).shape
    if dimension == (12,):
        return expand_active_frame(frame, dof_names)
    if dimension == (17,):
        return expand_independent_frame(frame, dof_names)
    raise ValueError(f"Linker单帧只能是12或17维，实际为{dimension}")


def longest_true_run(mask):
    """统计布尔序列中最长的连续真值长度。

    输入：一维布尔数组。
    输出：最长连续True步数整数。
    逻辑：单次扫描维护当前连续长度和历史最大值。
    作用：要求物体持续抬升，而非把分散的数值尖峰误判为成功。
    """
    best = current = 0
    for value in np.asarray(mask, dtype=bool):
        current = current + 1 if value else 0
        best = max(best, current)
    return int(best)
