"""构造分阶段物体接触计划，并计算目标手指腹区域的世界几何。

输入：Shadow轨迹/指尖、物体表面、目标手指腹配置和当前运动学模型。
输出：接近—闭合—抬升阶段、逐帧接触区域及可求梯度的目标手指腹点/法向。
内部逻辑：从源指尖距离与手腕抬升自动分段，抬升期让虚拟物体随源手腕刚体运动。
作用：把跨手可复用的接触计划与具体Linker/XHand优化器分离，保持单轨迹单候选。
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.optimize import nnls
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import torch


TIP_SEMANTICS = ("index", "middle", "ring", "little", "thumb")


def friction_wrench_residual(
    contact_points,
    outward_normals,
    object_center,
    friction_coefficient=1.0,
    cone_edges=4,
    torque_scale=1.0,
):
    """估计一组接触的摩擦锥能否平衡单位重力。

    输入：接触点、物体外法向、物体中心、摩擦系数、锥边数和力矩归一化尺度。
    输出：非负接触力组合的残差及各摩擦锥边权重；越接近0越能平衡重力和力矩。
    内部逻辑：把每个库仑摩擦锥离散成若干方向，构造`[力; 力矩]`抓取矩阵，
    再用非负最小二乘拟合抵消向下单位重力所需的向上力和零合力矩。
    作用：给接触区域选择加入最小物理承力判据，区别于只比较距离和法向夹角。
    """
    points = np.asarray(contact_points, dtype=np.float64)
    normals = np.asarray(outward_normals, dtype=np.float64)
    center = np.asarray(object_center, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or normals.shape != points.shape:
        raise ValueError("contact_points/outward_normals必须为相同的(N,3)")
    if friction_coefficient < 0 or cone_edges < 3 or torque_scale <= 0:
        raise ValueError("摩擦系数非负、锥边至少3、力矩尺度必须为正")
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    wrenches = []
    for point, normal in zip(points, normals):
        helper = np.asarray([1.0, 0.0, 0.0])
        if abs(float(np.dot(helper, normal))) > 0.9:
            helper = np.asarray([0.0, 1.0, 0.0])
        tangent1 = np.cross(normal, helper)
        tangent1 /= max(np.linalg.norm(tangent1), 1e-12)
        tangent2 = np.cross(normal, tangent1)
        for edge_index in range(int(cone_edges)):
            angle = 2.0 * np.pi * edge_index / int(cone_edges)
            tangent = np.cos(angle) * tangent1 + np.sin(angle) * tangent2
            force = -normal + float(friction_coefficient) * tangent
            force /= max(np.linalg.norm(force), 1e-12)
            lever = (point - center) / float(torque_scale)
            torque = np.cross(lever, force)
            wrenches.append(np.concatenate([force, torque]))
    grasp_matrix = np.stack(wrenches, axis=1)
    target = np.asarray([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    weights, residual = nnls(grasp_matrix, target)
    return float(residual), weights


def infer_motion_phases(
    source_frames,
    source_tip_points,
    object_vertices,
    contact_threshold,
    min_contact_tips,
    lift_delta,
    contact_fallback="error",
):
    """从专家指尖接近物体和手腕上升自动推断三段时序。

    输入：`(T,28)`源帧、五指世界点、物体顶点、三个阶段阈值及回退模式。
    输出：close/lift/grasp帧、逐指距离与接触掩码。
    内部逻辑：首次至少若干指尖入阈值为闭合开始；之后腕Z离低点上升指定距离为抬升。
    作用：替代对所有轨迹硬编码同一个第35帧，同时保持阶段定义可解释。
    """
    frames = np.asarray(source_frames, dtype=np.float64)
    if frames.ndim != 2 or frames.shape[1] < 6:
        raise ValueError(f"source_frames应为(T,D>=6)，实际为{frames.shape}")
    tree = cKDTree(np.asarray(object_vertices, dtype=np.float64))
    distances = {}
    for semantic in TIP_SEMANTICS:
        points = np.asarray(source_tip_points[semantic], dtype=np.float64)
        if points.shape != (len(frames), 3):
            raise ValueError(f"{semantic}源指尖形状错误: {points.shape}")
        distances[semantic] = tree.query(points, k=1)[0]
    distance_matrix = np.stack([distances[name] for name in TIP_SEMANTICS], axis=1)
    contact_mask = distance_matrix <= float(contact_threshold)
    counts = contact_mask.sum(axis=1)
    if contact_fallback not in {"error", "nearest"}:
        raise ValueError(f"未知contact_fallback: {contact_fallback}")
    if not 1 <= int(min_contact_tips) <= distance_matrix.shape[1]:
        raise ValueError(
            f"min_contact_tips必须在1..{distance_matrix.shape[1]}，"
            f"实际为{min_contact_tips}"
        )
    contact_order_distance = np.partition(
        distance_matrix, int(min_contact_tips) - 1, axis=1
    )[:, int(min_contact_tips) - 1]
    close_candidates = np.flatnonzero(counts >= int(min_contact_tips))
    fallback_used = not len(close_candidates)
    if fallback_used:
        if contact_fallback == "error":
            raise ValueError("专家轨迹中没有足够指尖接近物体，无法建立闭合阶段")
        # 第k近指尖的距离代表“同时让k根指尖接近”的最坏距离。
        # 取该距离最小的帧，比放宽成单指阈值更贴近原“多指闭合”定义。
        close_start = int(np.argmin(contact_order_distance))
        close_detection = "nearest_min_contact_tips"
    else:
        close_start = int(close_candidates[0])
        close_detection = "threshold"
    wrist_z = frames[:, 2]
    base_z = float(wrist_z[close_start:].min())
    lift_candidates = np.flatnonzero(
        (np.arange(len(frames)) >= close_start)
        & (wrist_z >= base_z + float(lift_delta))
    )
    if not len(lift_candidates):
        raise ValueError("专家轨迹没有达到指定手腕抬升量，无法建立抬升阶段")
    lift_start = int(lift_candidates[0])
    if fallback_used:
        # 回退帧本身就是多指同时最接近的帧，不再被单指入阈值帧覆盖。
        grasp_frame = close_start
    else:
        window_counts = counts[close_start : lift_start + 1]
        maximum = int(window_counts.max())
        maximum_frames = np.flatnonzero(window_counts == maximum) + close_start
        grasp_frame = int(maximum_frames[-1])
    return {
        "close_start_frame": close_start,
        "lift_start_frame": lift_start,
        "grasp_frame": grasp_frame,
        "close_detection": close_detection,
        "contact_fallback_used": bool(fallback_used),
        "close_contact_order_distance_m": float(
            contact_order_distance[close_start]
        ),
        "source_tip_distances_m": distance_matrix,
        "source_contact_mask": contact_mask,
        "source_contact_tip_count": counts,
    }


def wrist_rotation(frame):
    """把一帧Shadow XYZ欧拉角转成3×3腕部旋转矩阵。

    输入：至少6维的源姿态帧。
    输出：SciPy `xyz`约定的旋转矩阵。
    内部逻辑：读取第3:6维弧度欧拉角。
    作用：在抬升阶段让接触区域随专家手腕做刚体运动。
    """
    return Rotation.from_euler("xyz", np.asarray(frame)[3:6]).as_matrix()


def move_with_wrist(points, normals, base_frame, current_frame):
    """把基准世界表面随手腕相对运动变换到当前帧。

    输入：基准点/法向、基准Shadow腕姿态和当前腕姿态。
    输出：刚体移动后的世界点和单位法向。
    内部逻辑：先逆变换到基准腕坐标，再用当前腕姿态变回世界坐标。
    作用：抬升时使用“随手移动的虚拟物体”，避免把手错误拉回桌面上的初始物体。
    """
    points = np.asarray(points, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    rotation0, rotation1 = wrist_rotation(base_frame), wrist_rotation(current_frame)
    translation0 = np.asarray(base_frame[:3], dtype=np.float64)
    translation1 = np.asarray(current_frame[:3], dtype=np.float64)
    local_points = (points - translation0) @ rotation0
    moved_points = local_points @ rotation1.T + translation1
    local_normals = normals @ rotation0
    moved_normals = local_normals @ rotation1.T
    moved_normals /= np.maximum(np.linalg.norm(moved_normals, axis=1, keepdims=True), 1e-12)
    return moved_points, moved_normals


def nearest_surface_region(tree, vertices, normals, query_point, neighbors):
    """选取源语义指尖附近的一小片物体表面区域。

    输入：物体KD-tree、顶点/法向、源指尖点和邻居数。
    输出：邻近区域顶点及对应法向。
    内部逻辑：做k近邻查询并统一整理成一维索引。
    作用：比固定单点允许形态差异，又避免五指都滑向物体同一侧。
    """
    count = min(int(neighbors), len(vertices))
    _, indices = tree.query(np.asarray(query_point), k=count)
    indices = np.atleast_1d(indices).astype(np.int64)
    return vertices[indices], normals[indices]


def select_opposing_contact_regions(
    source_tip_points,
    vertices,
    normals,
    candidate_neighbors,
    region_neighbors,
    distance_scale,
    opposition_weight,
):
    """在专家语义邻域内联合选择拇指—四指对向接触区域。

    输入：单帧五指源点、物体表面、候选/输出邻域数、距离尺度和对向权重。
    输出：五指接触区域，以及选中中心的位移和拇指法向夹角诊断。
    内部逻辑：枚举拇指候选；对每个普通指寻找“离专家位置近且法向与拇指相反”的最佳点，
    再选择四指平均代价最低的共享拇指点。
    作用：避免各指独立最近点集中在同侧，使接触目标显式包含基本力闭合结构。
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    if distance_scale <= 0 or opposition_weight < 0:
        raise ValueError("distance_scale必须为正，opposition_weight不能为负")
    tree = cKDTree(vertices)
    candidate_count = min(int(candidate_neighbors), len(vertices))
    if candidate_count < 1:
        raise ValueError("candidate_neighbors必须为正整数")

    candidate_data = {}
    for semantic in TIP_SEMANTICS:
        distances, indices = tree.query(
            np.asarray(source_tip_points[semantic], dtype=np.float64),
            k=candidate_count,
        )
        candidate_data[semantic] = (
            np.atleast_1d(distances),
            np.atleast_1d(indices).astype(np.int64),
        )

    thumb_distances, thumb_indices = candidate_data["thumb"]
    thumb_normals = normals[thumb_indices]
    per_finger_best_cost = []
    per_finger_best_index = {}
    for semantic in TIP_SEMANTICS[:-1]:
        finger_distances, finger_indices = candidate_data[semantic]
        normal_dot = thumb_normals @ normals[finger_indices].T
        distance_cost = (
            thumb_distances[:, None] ** 2 + finger_distances[None, :] ** 2
        ) / float(distance_scale) ** 2
        pair_cost = distance_cost + float(opposition_weight) * (normal_dot + 1.0) ** 2
        best_for_each_thumb = np.argmin(pair_cost, axis=1)
        per_finger_best_index[semantic] = best_for_each_thumb
        per_finger_best_cost.append(
            pair_cost[np.arange(len(thumb_indices)), best_for_each_thumb]
        )
    shared_cost = np.mean(np.stack(per_finger_best_cost, axis=1), axis=1)
    chosen_thumb_candidate = int(np.argmin(shared_cost))
    chosen_indices = {"thumb": int(thumb_indices[chosen_thumb_candidate])}
    for semantic in TIP_SEMANTICS[:-1]:
        _, finger_indices = candidate_data[semantic]
        chosen_indices[semantic] = int(
            finger_indices[per_finger_best_index[semantic][chosen_thumb_candidate]]
        )

    regions, diagnostics = {}, {}
    thumb_normal = normals[chosen_indices["thumb"]]
    for semantic in TIP_SEMANTICS:
        center_index = chosen_indices[semantic]
        region_points, region_normals = nearest_surface_region(
            tree,
            vertices,
            normals,
            vertices[center_index],
            region_neighbors,
        )
        regions[semantic] = (region_points, region_normals)
        displacement = float(
            np.linalg.norm(vertices[center_index] - source_tip_points[semantic])
        )
        angle = None
        if semantic != "thumb":
            dot = float(np.clip(np.dot(thumb_normal, normals[center_index]), -1.0, 1.0))
            angle = float(np.degrees(np.arccos(dot)))
        diagnostics[semantic] = {
            "center_vertex_index": center_index,
            "source_tip_to_center_m": displacement,
            "thumb_normal_angle_deg": angle,
        }
    diagnostics["shared_normalized_cost"] = float(shared_cost[chosen_thumb_candidate])
    return regions, diagnostics


def select_reachable_opposing_contact_regions(
    reachable_pads,
    vertices,
    normals,
    candidate_neighbors,
    region_neighbors,
    distance_scale,
    opposition_weight,
    pad_alignment_weight,
    min_opposing_fingers=2,
    friction_stability_weight=0.0,
    friction_coefficient=1.0,
    friction_cone_edges=4,
    max_reachable_distance=0.0,
):
    """从目标手当前可达指腹附近选择拇指—四指对向接触区域。

    输入：五指基线指腹世界点/法向、物体表面、候选数量及距离/法向权重。
    输出：五指目标表面区域，以及可达距离、指腹法向和对向法向诊断。
    内部逻辑：先为每个物体顶点寻找距离最近且法向相对的真实指腹点；每指只保留
    代价最低的K个可达候选，再联合枚举拇指候选并为四指选择法向最对向的组合。
    作用：避免旧方法围绕Shadow指尖选择了O6无法到达的目标，显式尊重目标手形态。
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    if set(reachable_pads) != set(TIP_SEMANTICS):
        raise ValueError("reachable_pads必须恰好包含五根语义手指")
    if distance_scale <= 0 or opposition_weight < 0 or pad_alignment_weight < 0:
        raise ValueError("距离尺度必须为正，法向权重不能为负")
    if friction_stability_weight < 0 or friction_coefficient < 0:
        raise ValueError("摩擦稳定度权重和摩擦系数不能为负")
    if max_reachable_distance < 0:
        raise ValueError("最大可达距离不能为负")
    if not 1 <= int(min_opposing_fingers) <= 4:
        raise ValueError("min_opposing_fingers必须在1到4之间")
    candidate_count = min(int(candidate_neighbors), len(vertices))
    if candidate_count < 1:
        raise ValueError("candidate_neighbors必须为正整数")

    candidates = {}
    for semantic in TIP_SEMANTICS:
        pad_points, pad_normals = reachable_pads[semantic]
        pad_points = np.asarray(pad_points, dtype=np.float64)
        pad_normals = np.asarray(pad_normals, dtype=np.float64)
        if pad_points.ndim != 2 or pad_points.shape[1] != 3:
            raise ValueError(f"{semantic}指腹点形状错误: {pad_points.shape}")
        if pad_normals.shape != pad_points.shape:
            raise ValueError(f"{semantic}指腹法向形状错误: {pad_normals.shape}")
        pad_normals /= np.maximum(
            np.linalg.norm(pad_normals, axis=1, keepdims=True), 1e-12
        )
        difference = vertices[:, None, :] - pad_points[None, :, :]
        squared_distance = np.sum(difference * difference, axis=2)
        normal_dot = normals @ pad_normals.T
        pair_cost = (
            squared_distance / float(distance_scale) ** 2
            + float(pad_alignment_weight) * (normal_dot + 1.0) ** 2
        )
        best_pad = np.argmin(pair_cost, axis=1)
        best_cost = pair_cost[np.arange(len(vertices)), best_pad]
        order = np.argsort(best_cost, kind="stable")[:candidate_count]
        candidates[semantic] = {
            "indices": order,
            "cost": best_cost[order],
            "pad_indices": best_pad[order],
            "pad_points": pad_points,
            "pad_normals": pad_normals,
            "distance": np.sqrt(
                squared_distance[np.arange(len(vertices)), best_pad]
            )[order],
        }

    thumb = candidates["thumb"]
    thumb_indices = thumb["indices"]
    thumb_normals = normals[thumb_indices]
    finger_names = list(TIP_SEMANTICS[:-1])
    per_finger_choice, per_finger_cost = {}, []
    for semantic in TIP_SEMANTICS[:-1]:
        finger = candidates[semantic]
        opposition = (
            thumb_normals @ normals[finger["indices"]].T + 1.0
        ) ** 2
        pair_cost = (
            finger["cost"][None, :]
            + float(opposition_weight) * opposition
        )
        choice = np.argmin(pair_cost, axis=1)
        per_finger_choice[semantic] = choice
        per_finger_cost.append(pair_cost[np.arange(len(thumb_indices)), choice])
    finger_cost_matrix = np.stack(per_finger_cost, axis=1)
    selected_count = int(min_opposing_fingers)
    shared_cost = thumb["cost"] + np.mean(
        np.sort(finger_cost_matrix, axis=1)[:, :selected_count], axis=1
    )
    chosen_stability_residual = None
    if friction_stability_weight > 0:
        object_center = np.mean(vertices, axis=0)
        torque_scale = max(float(np.linalg.norm(np.ptp(vertices, axis=0))), 1e-6)
        best = None
        for candidate_thumb in range(len(thumb_indices)):
            if (
                max_reachable_distance > 0
                and thumb["distance"][candidate_thumb] > max_reachable_distance
            ):
                continue
            for finger_subset in combinations(range(len(finger_names)), selected_count):
                vertex_indices = [int(thumb_indices[candidate_thumb])]
                geometry_costs = []
                for finger_index in finger_subset:
                    semantic = finger_names[finger_index]
                    local_choice = int(
                        per_finger_choice[semantic][candidate_thumb]
                    )
                    if (
                        max_reachable_distance > 0
                        and candidates[semantic]["distance"][local_choice]
                        > max_reachable_distance
                    ):
                        vertex_indices = []
                        break
                    vertex_indices.append(
                        int(candidates[semantic]["indices"][local_choice])
                    )
                    geometry_costs.append(
                        finger_cost_matrix[candidate_thumb, finger_index]
                    )
                if not vertex_indices:
                    continue
                residual, _ = friction_wrench_residual(
                    vertices[vertex_indices],
                    normals[vertex_indices],
                    object_center,
                    friction_coefficient,
                    friction_cone_edges,
                    torque_scale,
                )
                geometry_cost = float(thumb["cost"][candidate_thumb]) + float(
                    np.mean(geometry_costs)
                )
                total_cost = geometry_cost + float(friction_stability_weight) * residual**2
                record = (total_cost, residual, candidate_thumb, finger_subset)
                if best is None or record[:2] < best[:2]:
                    best = record
        if best is None:
            raise ValueError("硬可达距离内没有满足手指数要求的摩擦接触组合")
        _, chosen_stability_residual, thumb_choice, selected_finger_indices = best
        selected_finger_indices = np.asarray(selected_finger_indices, dtype=np.int64)
    else:
        thumb_choice = int(np.argmin(shared_cost))
        selected_finger_indices = np.argsort(
            finger_cost_matrix[thumb_choice], kind="stable"
        )[:selected_count]
    chosen = {"thumb": thumb_choice}
    selected_fingers = [finger_names[index] for index in selected_finger_indices]
    for semantic in selected_fingers:
        chosen[semantic] = int(per_finger_choice[semantic][thumb_choice])

    tree = cKDTree(vertices)
    regions, diagnostics = {}, {}
    thumb_vertex = int(thumb_indices[thumb_choice])
    thumb_normal = normals[thumb_vertex]
    for semantic in ["thumb", *selected_fingers]:
        info = candidates[semantic]
        local_choice = chosen[semantic]
        vertex_index = int(info["indices"][local_choice])
        pad_index = int(info["pad_indices"][local_choice])
        regions[semantic] = nearest_surface_region(
            tree,
            vertices,
            normals,
            vertices[vertex_index],
            region_neighbors,
        )
        pad_point = info["pad_points"][pad_index]
        pad_normal = info["pad_normals"][pad_index]
        pad_dot = float(np.clip(np.dot(pad_normal, normals[vertex_index]), -1, 1))
        thumb_angle = None
        if semantic != "thumb":
            dot = float(np.clip(np.dot(thumb_normal, normals[vertex_index]), -1, 1))
            thumb_angle = float(np.degrees(np.arccos(dot)))
        diagnostics[semantic] = {
            "center_vertex_index": vertex_index,
            "reachable_pad_index": pad_index,
            "reachable_pad_to_center_m": float(
                np.linalg.norm(pad_point - vertices[vertex_index])
            ),
            "pad_object_normal_angle_deg": float(np.degrees(np.arccos(pad_dot))),
            "thumb_normal_angle_deg": thumb_angle,
        }
    diagnostics["shared_normalized_cost"] = float(shared_cost[thumb_choice])
    diagnostics["friction_wrench_residual"] = (
        None
        if chosen_stability_residual is None
        else float(chosen_stability_residual)
    )
    diagnostics["selector"] = "linker_reachable_pads"
    diagnostics["selected_opposing_fingers"] = selected_fingers
    return regions, diagnostics


def build_phase_contact_plan(
    source_frames,
    source_tip_points,
    object_vertices,
    object_normals,
    contact_threshold=0.02,
    min_contact_tips=2,
    lift_delta=0.03,
    region_neighbors=32,
    opposition_candidate_neighbors=0,
    opposition_distance_scale=0.03,
    opposition_weight=1.0,
    opposition_refine_frames=4,
    reachable_pads=None,
    reachable_pad_alignment_weight=1.0,
    reachable_min_opposing_fingers=2,
    friction_stability_weight=0.0,
    friction_coefficient=1.0,
    friction_cone_edges=4,
    max_reachable_distance=0.0,
    contact_fallback="error",
):
    """生成每帧静态或随腕移动的物体及五指语义接触区域。

    输入：源轨迹/指尖、初始物体表面、阶段/区域参数和可选对向选择参数。
    输出：阶段元数据与长度T的计划；每帧含物体代理和各指目标区域。
    内部逻辑：闭合期追随源指尖当前邻域；抬升期冻结grasp帧邻域并随腕刚体移动。
    作用：让目标手先靠近、再形成与专家同侧分布的接触、最后保持抓形抬升。
    """
    vertices = np.asarray(object_vertices, dtype=np.float64)
    normals = np.asarray(object_normals, dtype=np.float64)
    phases = infer_motion_phases(
        source_frames,
        source_tip_points,
        vertices,
        contact_threshold,
        min_contact_tips,
        lift_delta,
        contact_fallback,
    )
    tree = cKDTree(vertices)
    grasp_frame = phases["grasp_frame"]
    opposition_diagnostics = None
    if opposition_candidate_neighbors > 0:
        if reachable_pads is None:
            grasp_regions, opposition_diagnostics = select_opposing_contact_regions(
                {
                    semantic: source_tip_points[semantic][grasp_frame]
                    for semantic in TIP_SEMANTICS
                },
                vertices,
                normals,
                opposition_candidate_neighbors,
                region_neighbors,
                opposition_distance_scale,
                opposition_weight,
            )
        else:
            grasp_regions, opposition_diagnostics = (
                select_reachable_opposing_contact_regions(
                    reachable_pads,
                    vertices,
                    normals,
                    opposition_candidate_neighbors,
                    region_neighbors,
                    opposition_distance_scale,
                    opposition_weight,
                    reachable_pad_alignment_weight,
                    reachable_min_opposing_fingers,
                    friction_stability_weight,
                    friction_coefficient,
                    friction_cone_edges,
                    max_reachable_distance,
                )
            )
    else:
        if phases["contact_fallback_used"]:
            grasp_finger_indices = np.argsort(
                phases["source_tip_distances_m"][grasp_frame]
            )[: int(min_contact_tips)]
        else:
            grasp_finger_indices = np.flatnonzero(
                phases["source_contact_mask"][grasp_frame]
            )
        grasp_regions = {
            semantic: nearest_surface_region(
                tree,
                vertices,
                normals,
                source_tip_points[semantic][grasp_frame],
                region_neighbors,
            )
            for finger_index, semantic in enumerate(TIP_SEMANTICS)
            if finger_index in grasp_finger_indices
        }
    opposition_start = max(
        phases["close_start_frame"],
        phases["lift_start_frame"] - max(1, int(opposition_refine_frames)),
    )
    plan = []
    for frame_index, frame in enumerate(source_frames):
        if frame_index < phases["close_start_frame"]:
            plan.append(
                {"phase": "approach", "object_vertices": vertices, "object_normals": normals, "targets": {}}
            )
            continue
        if frame_index < phases["lift_start_frame"]:
            if phases["contact_fallback_used"]:
                plan.append(
                    {
                        "phase": "close",
                        "object_vertices": vertices,
                        "object_normals": normals,
                        "targets": grasp_regions,
                    }
                )
                continue
            if opposition_candidate_neighbors > 0 and frame_index >= opposition_start:
                plan.append(
                    {
                        "phase": "close",
                        "object_vertices": vertices,
                        "object_normals": normals,
                        "targets": grasp_regions,
                    }
                )
                continue
            targets = {}
            for finger_index, semantic in enumerate(TIP_SEMANTICS):
                if phases["source_contact_mask"][frame_index, finger_index]:
                    targets[semantic] = nearest_surface_region(
                        tree,
                        vertices,
                        normals,
                        source_tip_points[semantic][frame_index],
                        region_neighbors,
                    )
            plan.append(
                {"phase": "close", "object_vertices": vertices, "object_normals": normals, "targets": targets}
            )
            continue
        moved_vertices, moved_normals = move_with_wrist(
            vertices,
            normals,
            source_frames[phases["lift_start_frame"]],
            frame,
        )
        targets = {}
        for semantic, (region_points, region_normals) in grasp_regions.items():
            targets[semantic] = move_with_wrist(
                region_points,
                region_normals,
                source_frames[phases["lift_start_frame"]],
                frame,
            )
        plan.append(
            {"phase": "lift", "object_vertices": moved_vertices, "object_normals": moved_normals, "targets": targets}
        )
    return {
        **phases,
        "frames": plan,
        "opposition_start_frame": (
            opposition_start if opposition_candidate_neighbors > 0 else None
        ),
        "opposition_diagnostics": opposition_diagnostics,
    }


def load_pad_config(path, expected_hand=None):
    """读取并验证物理校准的五指区域配置。

    输入：JSON路径及可选目标手名称。
    输出：配置字典。
    内部逻辑：核对hand、五指齐全、每指点/法向数量和三维形状。
    作用：在昂贵逐帧优化前尽早阻止错手或不完整配置。
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if expected_hand is not None and data.get("hand") != expected_hand:
        raise ValueError(f"指腹配置属于{data.get('hand')}，期望{expected_hand}")
    if set(data.get("fingers", {})) != set(TIP_SEMANTICS):
        raise ValueError("指腹配置必须恰好包含五根语义手指")
    for semantic, info in data["fingers"].items():
        points = info.get("surface_points", [])
        if not points:
            raise ValueError(f"{semantic}没有表面点")
        for point in points:
            if np.asarray(point["local_xyz_m"]).shape != (3,):
                raise ValueError(f"{semantic}局部点不是3维")
            if np.asarray(point["local_outward_normal"]).shape != (3,):
                raise ValueError(f"{semantic}局部法向不是3维")
    return data


def world_pad_regions(model, pad_config):
    """将五指校准局部点/法向变换到当前可求梯度的世界坐标。

    输入：已调用`update_kinematics`的目标手模型和指腹配置。
    输出：语义手指到`(world_points, world_normals)`张量的字典。
    内部逻辑：先用link运动学变换，再应用模型全局腕旋转、平移和尺度。
    作用：让SLSQP能对真实指腹区域的距离与方向误差反向求导。
    """
    result = {}
    for semantic, info in pad_config["fingers"].items():
        local_points = torch.as_tensor(
            [item["local_xyz_m"] for item in info["surface_points"]],
            dtype=torch.float32,
            device=model.device,
        )
        local_normals = torch.as_tensor(
            [item["local_outward_normal"] for item in info["surface_points"]],
            dtype=torch.float32,
            device=model.device,
        )
        transform = model.current_status[info["body_name"]]
        hand_points = transform.transform_points(local_points)
        hand_normals = transform.transform_normals(local_normals)
        world_points = hand_points @ model.global_rotation[0].T + model.global_translation[0]
        world_normals = hand_normals @ model.global_rotation[0].T
        world_normals = world_normals / torch.clamp(
            torch.linalg.norm(world_normals, dim=1, keepdim=True), min=1e-8
        )
        result[semantic] = (world_points * float(model.scale), world_normals)
    return result


def pad_contact_terms(
    pads,
    contact_targets,
    object_surface,
    contact_offset=-0.001,
    min_signed_distance=-0.003,
):
    """计算跨目标手共享的指腹距离、法向和近似穿透基础项。

    输入：世界指腹点/法向、逐指物体区域、完整物体表面及两个有符号距离。
    输出：未乘方法权重的`contact/normal/penetration`标量张量字典。
    内部逻辑：每指选择距离最小的指腹—区域点对；穿透用最近顶点外法向近似符号。
    作用：让Linker与XHand使用完全相同的接触数学定义，避免复制后口径漂移。
    """
    contact_losses, normal_losses = [], []
    for semantic, (target_points_np, target_normals_np) in contact_targets.items():
        pad_points, pad_normals = pads[semantic]
        target_points = torch.as_tensor(
            target_points_np, dtype=torch.float32, device=pad_points.device
        )
        target_normals = torch.as_tensor(
            target_normals_np, dtype=torch.float32, device=pad_points.device
        )
        target_points = target_points + float(contact_offset) * target_normals
        pair_squared = torch.sum(
            (pad_points[:, None, :] - target_points[None, :, :]) ** 2, dim=2
        )
        flat_index = torch.argmin(pair_squared.detach()).item()
        pad_index = flat_index // pair_squared.shape[1]
        target_index = flat_index % pair_squared.shape[1]
        contact_losses.append(pair_squared[pad_index, target_index])
        dot = torch.sum(pad_normals[pad_index] * target_normals[target_index])
        normal_losses.append((dot + 1.0) ** 2)
    result = {}
    if contact_losses:
        result["contact"] = torch.mean(torch.stack(contact_losses))
        result["normal"] = torch.mean(torch.stack(normal_losses))
    if object_surface is not None:
        all_pad_points = torch.cat([points for points, _ in pads.values()], dim=0)
        object_points_np, object_normals_np = object_surface
        tree = cKDTree(np.asarray(object_points_np, dtype=np.float64))
        _, nearest_indices = tree.query(
            all_pad_points.detach().cpu().numpy(), k=1
        )
        nearest_points = torch.as_tensor(
            np.asarray(object_points_np)[nearest_indices],
            dtype=torch.float32,
            device=all_pad_points.device,
        )
        nearest_normals = torch.as_tensor(
            np.asarray(object_normals_np)[nearest_indices],
            dtype=torch.float32,
            device=all_pad_points.device,
        )
        signed_distance = torch.sum(
            (all_pad_points - nearest_points) * nearest_normals, dim=1
        )
        penetration = torch.relu(float(min_signed_distance) - signed_distance)
        result["penetration"] = torch.mean(penetration**2)
    return result
