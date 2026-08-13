#!/usr/bin/env python3
"""可视化Shadow与Linker O6的物理语义关键点对应。

输入：Linker语义配置、Shadow MJCF和Linker URDF/mesh。
输出：中性姿态交互HTML、三视图PNG和距离JSON。
内部逻辑：展开Linker的6主动+5 mimic运动学，将配置中的局部点变换到世界坐标。
作用：在写重定向优化器前确认新关键点确实位于正确指节和指尖。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import trimesh


RETARGET_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = RETARGET_ROOT.parent
REFERENCE_SCRIPTS = PROJECT_ROOT / "reference" / "HandRetargetTask2026" / "scripts"
THIRD_PARTY_PK = PROJECT_ROOT / "reference" / "HandRetargetTask2026" / "third_party" / "pytorch_kinematics"
for path in (REFERENCE_SCRIPTS, THIRD_PARTY_PK, Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mapping_visualization import (  # noqa: E402
    write_mapping_html,
    write_mapping_metrics,
    write_mapping_png,
)
from utils.hand_model import HandModel as ShadowHandModel  # noqa: E402
from utils.HandModel_linkerhand import HandModel_Linkerhand  # noqa: E402


def build_models():
    """创建CPU上的Shadow与Linker O6运动学模型。

    输入：无显式参数；读取考核参考仓库中的模型资产。
    输出：Shadow模型和带mimic展开的Linker模型。
    逻辑：Shadow启用21关键点；Linker只向外暴露6个主动关节。
    作用：提供两只手中性姿态正向运动学和mesh。
    """
    assets = REFERENCE_SCRIPTS / "assets"
    shadow_base = assets / "mjcf_free"
    shadow = ShadowHandModel(
        mjcf_path=str(shadow_base / "shadow_hand_vis_new.xml"),
        mesh_path=str(shadow_base / "meshes"),
        contact_points_path=str(shadow_base / "contact_points.json"),
        penetration_points_path=str(shadow_base / "penetration_points.json"),
        n_surface_points=0,
        device="cpu",
        use_joint21=True,
    )
    linker_asset = assets / "linkerhand" / "o6" / "right"
    linker = HandModel_Linkerhand(
        robot_name="linkerhand",
        urdf_filename="linkerhand_o6_right.urdf",
        mesh_path="",
        batch_size=1,
        device=torch.device("cpu"),
        mesh_nsp=128,
        hand_scale=1.0,
        asset_dir=str(linker_asset),
        allow_missing_contacts=True,
    )
    return shadow, linker


def neutral_poses():
    """构造与参考优化器一致的两只手中性姿态。

    输入：无。
    输出：Shadow 31维参数和Linker 15维参数（手腕9+主动关节6）。
    逻辑：关节置零，Linker使用参考旋转和`[0.003,0.002,-0.01]`平移。
    作用：在共同世界坐标系中检查物理关键点是否对应合理。
    """
    shadow_q = torch.zeros((1, 31), dtype=torch.float32)
    shadow_q[:, 3:9] = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    linker_q = torch.zeros((1, 15), dtype=torch.float32)
    linker_q[:, :3] = torch.tensor([0.003, 0.002, -0.01])
    rotation_alignment = torch.tensor(
        [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    linker_q[:, 3:9] = torch.cat(
        [rotation_alignment[:, 0], rotation_alignment[:, 1]], dim=0
    )
    return shadow_q, linker_q


def linker_world_point(linker, link_name, local_xyz):
    """把一个Linker link局部点转换到世界坐标。

    输入：已更新运动学的Linker模型、link名称和三维局部坐标。
    输出：形状`(3,)`的世界坐标NumPy数组。
    逻辑：先应用link变换，再应用手腕全局旋转、平移和手尺度。
    作用：让配置中的物理标志点无需修改参考关键点JSON即可参与检查。
    """
    local = torch.tensor(local_xyz, dtype=torch.float32).view(1, 1, 3)
    hand_point = linker.current_status[link_name].transform_points(local)[0, 0]
    world = hand_point @ linker.global_rotation[0].T + linker.global_translation[0]
    return (world * float(linker.scale)).detach().cpu().numpy()


def load_geometry():
    """计算Linker配置中全部语义点和双方mesh。

    输入：无显式参数；读取Linker映射配置和参考模型资产。
    输出：语义名称、双方匹配点、双方mesh和配置字典。
    逻辑：设置中性姿态，按每个pair的link与局部坐标计算世界点。
    作用：集中准备可视化和数值报告共同需要的数据。
    """
    config = json.loads(
        (RETARGET_ROOT / "configs" / "linker_o6_keypoint_map.json").read_text()
    )
    shadow, linker = build_models()
    shadow_q, linker_q = neutral_poses()
    shadow.set_parameters(shadow_q)
    linker.update_kinematics(linker_q)
    shadow_points = shadow.get_penetraion_keypoints()[0].detach().cpu().numpy()
    semantics = [pair["semantic"] for pair in config["pairs"]]
    source = np.stack(
        [shadow_points[pair["shadow_index"]] for pair in config["pairs"]]
    )
    target = np.stack(
        [
            linker_world_point(
                linker, pair["linker_link"], pair["linker_local_xyz"]
            )
            for pair in config["pairs"]
        ]
    )
    shadow_mesh = trimesh.util.concatenate(shadow.get_trimesh_data(0))
    linker_mesh = linker.get_meshes_from_q(q=linker_q, color=[0.2, 0.7, 0.2])
    return semantics, source, target, shadow_mesh, linker_mesh, config


def main():
    """执行Linker O6中性姿态关键点校准检查。

    输入：可选命令行`--output-dir`。
    输出：交互HTML、三视图PNG、距离JSON和终端摘要。
    逻辑：加载物理点与mesh，调用通用可视化函数写出三类产物。
    作用：作为Linker关键点进入优化器之前的prepare验收入口。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "linker_keypoint_check",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    semantics, source, target, shadow_mesh, linker_mesh, _ = load_geometry()
    html_path = args.output_dir / "neutral_mapping.html"
    png_path = args.output_dir / "neutral_mapping.png"
    metrics_path = args.output_dir / "neutral_mapping_metrics.json"
    write_mapping_html(
        html_path, "Linker O6", semantics, source, target, shadow_mesh, linker_mesh
    )
    write_mapping_png(
        png_path, "Linker O6", semantics, source, target, shadow_mesh, linker_mesh
    )
    report = write_mapping_metrics(
        metrics_path, "Linker O6", semantics, source, target
    )
    print(f"matched_pairs={report['pair_count']}")
    print(f"neutral_mean_distance_m={report['mean_distance_m']:.6f}")
    print(f"neutral_max_distance_m={report['max_distance_m']:.6f}")
    print(f"html={html_path}")
    print(f"png={png_path}")
    print(f"metrics={metrics_path}")


if __name__ == "__main__":
    main()

