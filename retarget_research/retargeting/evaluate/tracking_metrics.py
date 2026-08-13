"""计算位置控制命令与仿真实际关节状态之间的跟踪误差。

输入：逐物理步的实际DOF、命令DOF和自由度名称。
输出：逐自由度误差、误差时刻、数值范围和末步状态。
内部逻辑：仅用NumPy逐列统计，不依赖Isaac Gym。
作用：让控制执行误差可以独立测试，也避免离线工具被仿真依赖绑住。
"""

import numpy as np


def summarize_dof_tracking(actual, commanded, dof_names):
    """汇总位置命令与实际DOF状态之间的跟踪误差。

    输入：同形状逐步实际/命令位置数组和DOF名称。
    输出：逐DOF平均/最大绝对误差、最差步和实际/命令范围。
    内部逻辑：逐元素取绝对差后沿时间轴统计，不混合米制平移与弧度关节。
    作用：判断离线轨迹看似接触但物理执行时是否因PD滞后而没有到位。
    """
    actual = np.asarray(actual)
    commanded = np.asarray(commanded)
    if actual.ndim != 2 or commanded.ndim != 2:
        raise ValueError("DOF实际值和命令值必须是二维数组")
    if actual.shape != commanded.shape or actual.shape[1] != len(dof_names):
        raise ValueError("DOF实际值、命令值和名称数量不一致")
    absolute_error = np.abs(actual - commanded)
    maximum_by_dof = absolute_error.max(axis=0)
    maximum_step_by_dof = absolute_error.argmax(axis=0)
    worst_index = int(np.argmax(maximum_by_dof))
    return {
        "mean_absolute_tracking_error_by_dof": {
            name: float(absolute_error[:, index].mean())
            for index, name in enumerate(dof_names)
        },
        "max_absolute_tracking_error_by_dof": {
            name: float(maximum_by_dof[index])
            for index, name in enumerate(dof_names)
        },
        "max_tracking_error_step_by_dof": {
            name: int(maximum_step_by_dof[index])
            for index, name in enumerate(dof_names)
        },
        "commanded_position_range_by_dof": {
            name: [
                float(commanded[:, index].min()),
                float(commanded[:, index].max()),
            ]
            for index, name in enumerate(dof_names)
        },
        "actual_position_range_by_dof": {
            name: [float(actual[:, index].min()), float(actual[:, index].max())]
            for index, name in enumerate(dof_names)
        },
        "final_commanded_position_by_dof": {
            name: float(commanded[-1, index])
            for index, name in enumerate(dof_names)
        },
        "final_actual_position_by_dof": {
            name: float(actual[-1, index])
            for index, name in enumerate(dof_names)
        },
        "worst_tracking_dof": dof_names[worst_index],
        "worst_absolute_tracking_error": float(maximum_by_dof[worst_index]),
    }
