"""提供Wuji多候选选择与逐轨迹映射元数据读取的纯数据函数。

输入：候选评估指标或含单一/逐轨迹映射字段的npy字典。
输出：可排序的物理分数，以及某条轨迹实际使用的映射和语义。
内部逻辑：成功优先，其后比较持续抬升、最终/最大高度和接触步数。
作用：支持v1/v2混合候选，同时保持几何评估使用各轨迹自己的关键点定义。
"""

from pathlib import Path


def physics_selection_score(metrics):
    """把一条物理报告转换为确定性候选排序元组。

    输入：含成功、持续时间、抬升和接触步数的指标字典。
    输出：可直接按字典序取最大值的五元组。
    内部逻辑：成功标志最高优先级，再依次偏好持续更久、最终/最大更高和接触更多。
    作用：避免仅按单帧最大抬升选择短暂抓住后立即滑落的候选。
    """
    return (
        int(bool(metrics["success"])),
        float(metrics["longest_sustained_lift_time_s"]),
        float(metrics["final_lift_m"]),
        float(metrics["max_lift_m"]),
        int(metrics["hand_object_contact_steps"]),
    )


def trajectory_mapping_metadata(data, trajectory_index, default_mapping):
    """读取某条Wuji候选实际采用的映射文件和语义顺序。

    输入：候选字典、目标内部索引和缺省映射路径。
    输出：映射Path与语义字符串列表。
    内部逻辑：优先读取多候选文件的逐轨迹字段，否则退回单方法全局字段。
    作用：使v1/v2轨迹合并后仍能按各自生成目标做独立几何评估。
    """
    configs = data.get("mapping_config_per_trajectory")
    mapping = (
        data.get("mapping_config", default_mapping)
        if configs is None
        else configs[trajectory_index]
    )
    semantics_per_trajectory = data.get("mapping_semantics_per_trajectory")
    semantics = (
        data.get("mapping_semantics", [])
        if semantics_per_trajectory is None
        else semantics_per_trajectory[trajectory_index]
    )
    return Path(str(mapping)), [str(value) for value in semantics]
