"""把Isaac Gym接触记录转换为指腹校准所需的纯数据记录。

输入：Isaac风格结构化接触数组、手/物体刚体索引和时间索引。
输出：手部刚体局部接触点及环境接触法向的JSON兼容字典。
内部逻辑：识别手位于接触两侧中的哪一侧，并读取对应局部坐标字段。
作用：将不依赖仿真的数据规则隔离出来，便于单元测试和后续校准复用。
"""

from __future__ import annotations

import numpy as np


def vec3_list(value):
    """把Isaac Gym接触记录中的Vec3字段转换为普通三元素列表。

    输入：结构化NumPy记录中的`local_pos0/local_pos1/normal`字段。
    输出：可JSON序列化的三个浮点数。
    内部逻辑：兼容字段为`x/y/z`结构或普通长度3数组的两种Isaac表示。
    作用：隔离底层接触数据格式，避免校准脚本依赖特定Isaac版本。
    """
    names = getattr(getattr(value, "dtype", None), "names", None)
    if names and all(axis in names for axis in ("x", "y", "z")):
        return [float(value[axis]) for axis in ("x", "y", "z")]
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != 3:
        raise ValueError(f"接触Vec3字段应有3个元素，实际为{array.shape}")
    return array.tolist()


def collect_hand_object_local_contacts(
    contacts,
    hand_index_to_name,
    object_body_indices,
    physics_step,
    trajectory_frame,
):
    """提取手—物接触在手部刚体局部坐标系中的位置。

    输入：当前步接触记录、手刚体索引到名称、物体索引及当前步/轨迹帧编号。
    输出：每个手物接触对应的名称、局部坐标、环境法向和时刻记录列表。
    内部逻辑：识别手位于body0还是body1，并选择同侧的`local_pos`字段。
    作用：从成功物理重放中反推出真实指腹接触区域，供接触目标校准使用。
    """
    field_names = set(getattr(contacts.dtype, "names", ()) or ())

    def resolve_field(snake_name, camel_name):
        """从文档式和NumPy绑定式名称中选择当前实际存在的字段。"""
        if snake_name in field_names:
            return snake_name
        if camel_name in field_names:
            return camel_name
        raise ValueError(
            f"Isaac接触记录缺少{snake_name}/{camel_name}，实际字段={sorted(field_names)}"
        )

    local_pos0 = resolve_field("local_pos0", "localPos0")
    local_pos1 = resolve_field("local_pos1", "localPos1")
    object_indices = set(object_body_indices)
    samples = []
    for contact in contacts:
        body0, body1 = int(contact["body0"]), int(contact["body1"])
        if body0 in hand_index_to_name and body1 in object_indices:
            hand_index, local_field, hand_side = body0, local_pos0, 0
        elif body1 in hand_index_to_name and body0 in object_indices:
            hand_index, local_field, hand_side = body1, local_pos1, 1
        else:
            continue
        samples.append(
            {
                "physics_step": int(physics_step),
                "trajectory_frame": int(trajectory_frame),
                "hand_body": hand_index_to_name[hand_index],
                "hand_body_side": hand_side,
                "hand_local_position_m": vec3_list(contact[local_field]),
                "environment_contact_normal": vec3_list(contact["normal"]),
            }
        )
    return samples
