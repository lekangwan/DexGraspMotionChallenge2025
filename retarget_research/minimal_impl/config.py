"""集中保存三只手的维度、资产位置和语义关键点。

输入：无运行时输入，路径都相对当前仓库解析。
输出：其余模块共享的常量和 ``HandSpec``。
逻辑：把原实验中分散在多个JSON和脚本里的最终核心配置合并到一处。
作用：阅读者先从本文件理解三只手的差异，不必在几十个配置中跳转。
"""

from dataclasses import dataclass
from pathlib import Path


# 路径部分：代码可以从仓库任意工作目录启动。
ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "retarget_research/reference/HandRetargetTask2026"
REFERENCE_SCRIPTS = REFERENCE / "scripts"
REFERENCE_PK = REFERENCE / "third_party/pytorch_kinematics"


@dataclass(frozen=True)
class HandSpec:
    """描述一只目标手。

    输入：名称、手指动作维度、Isaac物理DOF数和运动学关键点索引。
    输出：不可变配置对象。
    逻辑：源索引与目标索引按相同顺序组成语义对应。
    作用：让同一重定向和仿真代码支持三种不同构型。
    """

    name: str
    finger_dim: int
    physics_dof: int
    source_indices: tuple
    target_indices: tuple

    @property
    def command_dim(self):
        """返回保存动作维度；输入无，输出为手腕6维加手指维度。"""
        return 6 + self.finger_dim


# 15点顺序：掌心，四指各近端/中段/指尖，拇指中段/指尖。
SOURCE_15 = (0, 1, 3, 4, 5, 7, 8, 9, 11, 12, 13, 15, 16, 19, 20)
XHAND_15 = (0, 10, 12, 14, 15, 17, 19, 20, 22, 24, 25, 27, 29, 6, 8)
WUJI_15 = (0, 6, 8, 10, 11, 13, 15, 16, 18, 20, 21, 23, 25, 3, 5)

HANDS = {
    "linker": HandSpec("linker", 6, 17, SOURCE_15, tuple(range(15))),
    "xhand": HandSpec("xhand", 12, 18, SOURCE_15, XHAND_15),
    "wuji": HandSpec("wuji", 20, 26, SOURCE_15, WUJI_15),
}


# Linker的15个局部点。它们是URDF关节中心和指尖mesh上校准后的点。
LINKER_POINTS = (
    ("rh_hand_base_link", (0.0, 0.0, 0.0)),
    ("rh_index_proximal", (-0.0052516, 0.0, 0.036625)),
    ("rh_index_distal", (0.005424345, 0.000002528, 0.013901684)),
    ("rh_index_distal", (0.014660391, 0.000006831, 0.037572119)),
    ("rh_middle_proximal", (-0.0052516, 0.0, 0.036625)),
    ("rh_middle_distal", (0.005424345, 0.000002528, 0.013901684)),
    ("rh_middle_distal", (0.014660391, 0.000006831, 0.037572119)),
    ("rh_ring_proximal", (-0.0052516, 0.0, 0.036625)),
    ("rh_ring_distal", (0.005424345, 0.000002528, 0.013901684)),
    ("rh_ring_distal", (0.014660391, 0.000006831, 0.037572119)),
    ("rh_pinky_proximal", (-0.0052516, 0.0, 0.036625)),
    ("rh_pinky_distal", (0.005424345, 0.000002528, 0.013901684)),
    ("rh_pinky_distal", (0.014660391, 0.000006831, 0.037572119)),
    ("rh_thumb_metacarpals", (0.0037776, 0.0, 0.045368)),
    ("rh_thumb_distal", (-0.006392941, 0.0, 0.048517395)),
)


# Linker功能向量：前10项匹配掌心到关节点，后5项匹配手指朝向。
LINKER_VECTORS = (
    ("position", 0, 4, 0, 3, 1.0),
    ("position", 0, 8, 0, 6, 1.0),
    ("position", 0, 12, 0, 9, 1.0),
    ("position", 0, 16, 0, 12, 1.0),
    ("position", 0, 20, 0, 14, 1.0),
    ("position", 0, 1, 0, 1, 0.5),
    ("position", 0, 5, 0, 4, 0.5),
    ("position", 0, 9, 0, 7, 0.5),
    ("position", 0, 13, 0, 10, 0.5),
    ("position", 0, 19, 0, 13, 0.5),
    ("direction", 1, 4, 1, 3, 5.0),
    ("direction", 5, 8, 4, 6, 5.0),
    ("direction", 9, 12, 7, 9, 5.0),
    ("direction", 13, 16, 10, 12, 5.0),
    ("direction", 19, 20, 13, 14, 5.0),
)

# 坐标系对齐矩阵
R_ALIGN = ((0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0))

# linker的抓握偏置，用于loss惩罚握得不够紧
LINKER_GRIP_TARGET = (1.16, 0.58, 0.94, 1.26, 1.46, 1.54)


# Isaac中的动作名称。名称映射比依赖隐式数组下标更容易读懂。
# linker添加的手腕虚拟的六个自由度
WRIST_LINKER_WUJI = (
    "virtual_joint_x",
    "virtual_joint_y",
    "virtual_joint_z",
    "virtual_joint_roll",
    "virtual_joint_pitch",
    "virtual_joint_yaw",
)
WRIST_XHAND = (
    "virtual_x",
    "virtual_y",
    "virtual_z",
    "virtual_rx",
    "virtual_ry",
    "virtual_rz",
)
XHAND_FINGERS = (
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
)
