#!/usr/bin/env python3
"""可视化Shadow与Wuji右手的15对初始语义关键点。

输入：Wuji映射JSON、Shadow MJCF及Wuji右手URDF/mesh/26点文件。
输出：中性姿态交互HTML、三视图PNG和逐点距离JSON。
内部逻辑：两手关节置零并应用共同坐标旋转，选出15对点后调用通用绘图函数。
作用：人工确认finger1到finger5、掌心和各指节位置，再决定是否修改映射。
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
THIRD_PARTY_PK = (
    PROJECT_ROOT
    / "reference"
    / "HandRetargetTask2026"
    / "third_party"
    / "pytorch_kinematics"
)
for path in (REFERENCE_SCRIPTS, THIRD_PARTY_PK, Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mapping_visualization import (  # noqa: E402
    write_mapping_html,
    write_mapping_metrics,
    write_mapping_png,
)
from utils.HandModel_xhand import HandModel_xhand  # noqa: E402
from utils.hand_model import HandModel as ShadowHandModel  # noqa: E402


def build_models():
    """创建用于可视化的Shadow和20关节Wuji CPU模型。

    输入：只读参考仓库中的两只手资产。
    输出：21点Shadow模型和26点Wuji模型。
    内部逻辑：关闭Shadow表面采样，Wuji允许缺少当前不需要的contact文件。
    作用：提供中性姿态关键点和完整三角网格。
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
    wuji_asset = assets / "wujihand_urdf" / "urdf"
    wuji = HandModel_xhand(
        robot_name="wuji_right",
        urdf_filename="right.urdf",
        mesh_path="../",
        batch_size=1,
        device=torch.device("cpu"),
        mesh_nsp=128,
        hand_scale=1.0,
        asset_dir=str(wuji_asset),
        allow_missing_contacts=True,
    )
    return shadow, wuji


def neutral_poses(wuji):
    """构造与Wuji首帧初始化相同的双方零关节姿态。

    输入：Wuji模型，用于读取20个关节。
    输出：Shadow 31维和Wuji 29维模型参数。
    内部逻辑：关节/平移置零，Wuji手腕应用固定Shadow→目标手坐标旋转。
    作用：在不受某条优化结果影响的情况下先检查映射语义与手型方向。
    """
    shadow_q = torch.zeros((1, 31), dtype=torch.float32)
    shadow_q[:, 3:9] = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    joint_count = len(wuji.robot.get_joint_parameter_names())
    wuji_q = torch.zeros((1, 9 + joint_count), dtype=torch.float32)
    alignment = torch.tensor(
        [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    wuji_q[:, 3:9] = torch.cat([alignment[:, 0], alignment[:, 1]], dim=0)
    return shadow_q, wuji_q


def load_geometry(mapping_config):
    """取得中性姿态的15对点和双方完整mesh。

    输入：Wuji映射配置路径及两只手参考资产。
    输出：语义列表、双方选中点、双方mesh和配置。
    内部逻辑：执行正向运动学后按`shadow_index/wuji_index`选择相同语义。
    作用：集中准备HTML、PNG和距离报告共享的数据。
    """
    config = json.loads(
        Path(mapping_config).read_text()
    )
    shadow, wuji = build_models()
    shadow_q, wuji_q = neutral_poses(wuji)
    shadow.set_parameters(shadow_q)
    shadow_points = shadow.get_penetraion_keypoints()[0].detach().numpy()
    wuji_points = wuji.get_penetraion_keypoints(q=wuji_q)[0].detach().numpy()
    semantics = [pair["semantic"] for pair in config["pairs"]]
    source = np.stack(
        [shadow_points[pair["shadow_index"]] for pair in config["pairs"]]
    )
    target = np.stack(
        [wuji_points[pair["wuji_index"]] for pair in config["pairs"]]
    )
    shadow_mesh = trimesh.util.concatenate(shadow.get_trimesh_data(0))
    wuji_mesh = wuji.get_meshes_from_q(q=wuji_q, color=[0.2, 0.7, 0.2])
    return semantics, source, target, shadow_mesh, wuji_mesh, config


def main():
    """运行Wuji中性姿态映射可视化检查。

    输入：可选`--output-dir`，默认写入项目outputs。
    输出：`neutral_mapping.html/png`和距离JSON及终端摘要。
    内部逻辑：加载几何并调用三种通用输出函数。
    作用：作为Wuji关键点映射进入下一轮优化前的人工验收入口。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "wuji_keypoint_check",
    )
    parser.add_argument(
        "--mapping-config",
        type=Path,
        default=RETARGET_ROOT / "configs" / "wuji_keypoint_map.json",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    semantics, source, target, shadow_mesh, wuji_mesh, _ = load_geometry(
        args.mapping_config
    )
    html_path = args.output_dir / "neutral_mapping.html"
    png_path = args.output_dir / "neutral_mapping.png"
    metrics_path = args.output_dir / "neutral_mapping_metrics.json"
    write_mapping_html(
        html_path, "Wuji", semantics, source, target, shadow_mesh, wuji_mesh
    )
    write_mapping_png(
        png_path, "Wuji", semantics, source, target, shadow_mesh, wuji_mesh
    )
    report = write_mapping_metrics(
        metrics_path, "Wuji", semantics, source, target
    )
    print(f"matched_pairs={report['pair_count']}")
    print(f"neutral_mean_distance_m={report['mean_distance_m']:.6f}")
    print(f"neutral_max_distance_m={report['max_distance_m']:.6f}")
    print(f"html={html_path}")
    print(f"png={png_path}")
    print(f"metrics={metrics_path}")


if __name__ == "__main__":
    main()
