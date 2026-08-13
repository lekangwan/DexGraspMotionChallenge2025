#!/usr/bin/env python3
"""从成功物理重放中校准目标手的真实指腹表面区域。

输入：目标手类型、若干含局部接触样本的成功重放JSON和输出路径。
输出：每根手指的代表性局部表面点/外法向JSON，以及五指网格校准PNG。
内部逻辑：仅保留掌侧远端刚体，去重后做确定性k-medoids压缩，再投影到STL顶点。
作用：用真实稳定抓取接触替代link原点猜测，为接触距离、法向和穿透损失提供依据。
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
import trimesh


RETARGET_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_SCRIPTS = (
    RETARGET_ROOT.parent / "reference" / "HandRetargetTask2026" / "scripts"
)
XHAND_MESH_ROOT = REFERENCE_SCRIPTS / "assets" / "xhand" / "meshes"
LINKER_MESH_ROOT = (
    REFERENCE_SCRIPTS / "assets" / "linkerhand" / "o6" / "right" / "meshes"
)
WUJI_MESH_ROOT = (
    REFERENCE_SCRIPTS / "assets" / "wujihand_urdf" / "meshes" / "right"
)

HAND_SPECS = {
    "xhand": {
        "index": ("right_hand_index_rota_link2", "right_hand_index_rota_link2.STL"),
        "middle": ("right_hand_mid_link2", "right_hand_mid_link2.STL"),
        "ring": ("right_hand_ring_link2", "right_hand_ring_link2.STL"),
        "little": ("right_hand_pinky_link2", "right_hand_pinky_link2.STL"),
        "thumb": ("right_hand_thumb_rota_link2", "right_hand_thumb_rota_link2.STL"),
    },
    "linker": {
        "index": ("rh_index_distal", "index_distal.STL"),
        "middle": ("rh_middle_distal", "middle_distal.STL"),
        "ring": ("rh_ring_distal", "ring_distal.STL"),
        "little": ("rh_pinky_distal", "pinky_distal.STL"),
        "thumb": ("rh_thumb_distal", "thumb_distal.STL"),
    },
    "wuji": {
        "index": ("finger2_tip_link", "finger2_tip_link.STL"),
        "middle": ("finger3_tip_link", "finger3_tip_link.STL"),
        "ring": ("finger4_tip_link", "finger4_tip_link.STL"),
        "little": ("finger5_tip_link", "finger5_tip_link.STL"),
        "thumb": ("finger1_tip_link", "finger1_tip_link.STL"),
    },
}


def mesh_root(hand):
    """返回目标手STL根目录。

    输入：`xhand`、`linker`或`wuji`。
    输出：只读参考资产中的mesh目录Path。
    内部逻辑：按手类型查询固定资产位置。
    作用：保证接触点和优化运动学使用同一URDF配套几何。
    """
    roots = {
        "xhand": XHAND_MESH_ROOT,
        "linker": LINKER_MESH_ROOT,
        "wuji": WUJI_MESH_ROOT,
    }
    return roots[hand]


def unique_points(points, resolution_m=0.0005):
    """把物理步中大量重复接触压缩到固定空间分辨率。

    输入：`(N,3)`局部接触坐标和默认0.5 mm分辨率。
    输出：按量化格去重后的真实样本点。
    内部逻辑：坐标除以分辨率后取整，同一格仅保留第一条。
    作用：避免持续时间更长的同一接触位置支配指腹区域校准。
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"接触点应为(N,3)，实际为{points.shape}")
    cells = np.round(points / float(resolution_m)).astype(np.int64)
    _, indices = np.unique(cells, axis=0, return_index=True)
    return points[np.sort(indices)]


def deterministic_medoids(points, count):
    """用确定性k-medoids选取仍属于原样本的代表点。

    输入：去重后的`(N,3)`点和期望代表点数。
    输出：最多`count`个真实样本点及各自簇支持数。
    内部逻辑：以全局中心附近点起步，最远点初始化，再交替分配和更新medoid。
    作用：覆盖指腹不同位置，同时避免普通均值落进手指网格内部。
    """
    points = np.asarray(points, dtype=np.float64)
    count = min(int(count), len(points))
    if count < 1:
        raise ValueError("至少需要一个接触点")
    center = np.median(points, axis=0)
    selected = [int(np.argmin(np.linalg.norm(points - center, axis=1)))]
    while len(selected) < count:
        distance = np.min(
            np.linalg.norm(points[:, None, :] - points[selected][None, :, :], axis=2),
            axis=1,
        )
        distance[selected] = -1.0
        selected.append(int(np.argmax(distance)))
    medoids = points[selected].copy()
    labels = np.zeros(len(points), dtype=np.int64)
    for _ in range(20):
        labels = np.argmin(
            np.linalg.norm(points[:, None, :] - medoids[None, :, :], axis=2), axis=1
        )
        updated = medoids.copy()
        for index in range(count):
            cluster = points[labels == index]
            if len(cluster):
                cluster_center = np.median(cluster, axis=0)
                updated[index] = cluster[
                    np.argmin(np.linalg.norm(cluster - cluster_center, axis=1))
                ]
        if np.allclose(updated, medoids):
            break
        medoids = updated
    labels = np.argmin(
        np.linalg.norm(points[:, None, :] - medoids[None, :, :], axis=2), axis=1
    )
    support = np.bincount(labels, minlength=count)
    order = np.lexsort((medoids[:, 2], medoids[:, 1], medoids[:, 0]))
    return medoids[order], support[order]


def project_to_mesh(mesh, points):
    """把接触代表点投影到原始STL表面并取得局部外法向。

    输入：Trimesh和`(K,3)`接触代表点。
    输出：最近mesh顶点、对应顶点外法向和投影距离。
    内部逻辑：用KD-tree查询最近顶点并将其法向单位化。
    作用：消除PhysX接触offset/凸分解带来的毫米级离面误差。
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
    distances, indices = cKDTree(vertices).query(np.asarray(points, dtype=np.float64))
    selected_normals = normals[indices]
    selected_normals /= np.maximum(
        np.linalg.norm(selected_normals, axis=1, keepdims=True), 1e-12
    )
    return vertices[indices], selected_normals, np.asarray(distances)


def load_contact_evidence(report_paths, hand):
    """读取成功报告并按五根目标手掌侧远端刚体汇总接触点。

    输入：JSON路径列表和目标手类型。
    输出：刚体名称到局部点列表的字典，以及可追溯报告摘要。
    内部逻辑：拒绝失败或缺少详细样本的报告，只选择HAND_SPECS中的主指腹刚体。
    作用：防止掌心、近端或指背的偶然碰撞污染指腹定义。
    """
    allowed = {body for body, _ in HAND_SPECS[hand].values()}
    grouped = defaultdict(list)
    evidence = []
    for path in report_paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not data.get("success"):
            raise ValueError(f"校准只接受成功重放: {path}")
        samples = data.get("hand_object_local_contact_samples")
        if samples is None:
            raise ValueError(f"报告缺少--include-contact-samples数据: {path}")
        selected_count = 0
        for sample in samples:
            if sample["hand_body"] in allowed:
                grouped[sample["hand_body"]].append(sample["hand_local_position_m"])
                selected_count += 1
        evidence.append(
            {
                "report": str(path.resolve()),
                "object_name": data["object_name"],
                "source_trajectory_index": data["source_trajectory_index"],
                "all_hand_object_contacts": len(samples),
                "selected_pad_contacts": selected_count,
            }
        )
    return grouped, evidence


def calibrate_hand(hand, report_paths, points_per_finger):
    """完成五指区域压缩、网格投影并生成配置字典。

    输入：目标手、成功重放列表和每指代表点数量。
    输出：含证据、每指刚体/mesh、表面点与法向的完整配置。
    内部逻辑：逐指去重、medoid压缩、STL投影并记录样本覆盖范围。
    作用：形成后续单次物体感知重定向的可审计固定输入。
    """
    grouped, evidence = load_contact_evidence(report_paths, hand)
    fingers = {}
    for semantic, (body, mesh_name) in HAND_SPECS[hand].items():
        raw = np.asarray(grouped[body], dtype=np.float64)
        if len(raw) == 0:
            raise ValueError(f"成功样本中{semantic}/{body}没有接触，无法校准")
        unique = unique_points(raw)
        medoids, support = deterministic_medoids(unique, points_per_finger)
        mesh_path = mesh_root(hand) / mesh_name
        mesh = trimesh.load_mesh(mesh_path, process=False)
        surface, normals, projection = project_to_mesh(mesh, medoids)
        fingers[semantic] = {
            "body_name": body,
            "mesh_path": str(mesh_path.resolve()),
            "raw_contact_sample_count": int(len(raw)),
            "unique_0p5mm_cell_count": int(len(unique)),
            "observed_local_bounds_m": [raw.min(axis=0).tolist(), raw.max(axis=0).tolist()],
            "surface_points": [
                {
                    "local_xyz_m": point.tolist(),
                    "local_outward_normal": normal.tolist(),
                    "cluster_unique_cell_support": int(cluster_support),
                    "contact_to_mesh_projection_m": float(distance),
                }
                for point, normal, cluster_support, distance in zip(
                    surface, normals, support, projection
                )
            ],
        }
    return {
        "status": "physics_contact_calibrated_development_v1",
        "hand": hand,
        "calibration_rule": (
            "仅使用严格成功重放；每0.5mm去重；每指确定性k-medoids；"
            "投影到配套STL最近顶点并读取局部外法向"
        ),
        "points_per_finger": int(points_per_finger),
        "evidence": evidence,
        "fingers": fingers,
    }


def render_calibration(config, output_path):
    """渲染五根独立mesh上的指腹点和外法向箭头。

    输入：校准配置和PNG输出路径。
    输出：一行五列的三维网格/红点/蓝色法向图。
    内部逻辑：对大mesh稀疏绘制三角面，并以统一相对长度画局部法向。
    作用：让人工审查确认代表点确实位于预期掌侧远端表面。
    """
    figure = plt.figure(figsize=(18, 4))
    for plot_index, (semantic, info) in enumerate(config["fingers"].items(), start=1):
        axis = figure.add_subplot(1, 5, plot_index, projection="3d")
        mesh = trimesh.load_mesh(info["mesh_path"], process=False)
        faces = np.asarray(mesh.faces)
        stride = max(1, len(faces) // 1200)
        axis.plot_trisurf(
            mesh.vertices[:, 0],
            mesh.vertices[:, 1],
            mesh.vertices[:, 2],
            triangles=faces[::stride],
            color="#cccccc",
            alpha=0.25,
            linewidth=0.05,
        )
        points = np.asarray([item["local_xyz_m"] for item in info["surface_points"]])
        normals = np.asarray(
            [item["local_outward_normal"] for item in info["surface_points"]]
        )
        extent = np.ptp(np.asarray(mesh.vertices), axis=0).max()
        axis.scatter(points[:, 0], points[:, 1], points[:, 2], c="red", s=28)
        axis.quiver(
            points[:, 0], points[:, 1], points[:, 2],
            normals[:, 0], normals[:, 1], normals[:, 2],
            length=0.18 * extent, color="blue", normalize=True,
        )
        axis.set_title(f"{semantic}\n{info['raw_contact_sample_count']} contacts")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_zlabel("z")
        axis.set_box_aspect(np.maximum(np.ptp(np.asarray(mesh.vertices), axis=0), 1e-4))
    figure.suptitle(f"{config['hand']} physics-calibrated contact pads")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main():
    """解析参数、执行校准并保存JSON/PNG。

    输入：`--hand`、一个或多个`--report`、`--output`和`--plot`。
    输出：终端逐指样本摘要以及两个校准文件。
    内部逻辑：固定报告排序后调用校准与渲染函数。
    作用：作为prepare阶段从物理证据生成接触优化配置的标准入口。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand", choices=sorted(HAND_SPECS), required=True)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--points-per-finger", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", type=Path, required=True)
    args = parser.parse_args()
    config = calibrate_hand(args.hand, sorted(args.report), args.points_per_finger)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    render_calibration(config, args.plot)
    for semantic, info in config["fingers"].items():
        mean_projection = np.mean(
            [point["contact_to_mesh_projection_m"] for point in info["surface_points"]]
        )
        print(
            f"{semantic}: raw={info['raw_contact_sample_count']} "
            f"unique={info['unique_0p5mm_cell_count']} "
            f"mean_projection_mm={mean_projection * 1000:.3f}"
        )
    print(f"output={args.output}")
    print(f"plot={args.plot}")


if __name__ == "__main__":
    main()
