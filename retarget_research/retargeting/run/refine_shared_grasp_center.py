#!/usr/bin/env python3
"""为Linker、XHand和Wuji统一生成抓取中心腕部校准候选。

输入：手类型、冻结manifest、既有目标轨迹、中心目标类型、最大修正毫米数和输出目录。
输出：与manifest对齐的单候选npy，以及逐轨迹阶段、中心和动作不变量审计。
内部逻辑：由目标手正向运动学计算抬升首帧五指中心；目标可选物体世界AABB
中心或同帧Shadow专家五指中心；闭合期间渐进加入受限平移，lift后保持常量。
作用：用同一可解释规则比较三种不同结构的手，避免为每只手单独发明固定XYZ偏移。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch


RUN_DIR = Path(__file__).resolve().parent
RETARGET_ROOT = RUN_DIR.parent
PREPARE_DIR = RETARGET_ROOT / "prepare"
EVALUATE_DIR = RETARGET_ROOT / "evaluate"
for path in (RUN_DIR, PREPARE_DIR, EVALUATE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import evaluate_linker_geometry as linker_geometry  # noqa: E402
import evaluate_wuji_geometry as wuji_geometry  # noqa: E402
import evaluate_xhand_geometry as xhand_geometry  # noqa: E402
from object_geometry import transformed_object_vertices  # noqa: E402
from phase_contact import infer_motion_phases  # noqa: E402
from refine_linker_object_centric_advance import (  # noqa: E402
    TIP_SEMANTICS,
    apply_object_centric_advance,
)
from slice_manifest_candidates import slice_candidate  # noqa: E402
from wuji_candidate_utils import trajectory_mapping_metadata  # noqa: E402


HAND_DIMENSIONS = {"linker": 12, "xhand": 18, "wuji": 26}
SHADOW_TIP_INDICES = {
    "index": 4,
    "middle": 8,
    "ring": 12,
    "little": 16,
    "thumb": 20,
}


def build_hand_models(hand: str):
    """创建指定手及其共享Shadow源手的CPU运动学模型。

    输入：`linker/xhand/wuji`之一。
    输出：Shadow模型和对应目标手模型。
    内部逻辑：复用各手独立几何评估器的固定资产与模型构造函数。
    作用：保证新后处理看到的关键点和正式几何评测使用完全相同的定义。
    """
    if hand == "linker":
        return linker_geometry.build_models("coupled6")
    if hand == "xhand":
        return xhand_geometry.build_models()
    if hand == "wuji":
        return wuji_geometry.build_models()
    raise ValueError(f"未知目标手: {hand}")


def shadow_points_and_phases(
    shadow_model,
    source_frames: np.ndarray,
    object_vertices: np.ndarray,
    contact_threshold: float,
    min_contact_tips: int,
    lift_delta: float,
) -> tuple[np.ndarray, dict]:
    """计算完整Shadow关键点并从专家接触推断闭合/抬升阶段。

    输入：Shadow模型、已加Z偏移的28维轨迹、物体表面和三项阶段阈值。
    输出：`(70,21,3)`世界关键点及统一阶段字典。
    内部逻辑：批量正向运动学后抽取五个固定语义指尖，调用共享阶段检测；
    无双指入阈值时使用第k近指尖距离最小帧作为显式回退。
    作用：给本身没有阶段元数据的XHand/Wuji候选建立相同闭合时间口径。
    """
    with torch.no_grad():
        shadow_model.set_parameters(xhand_geometry.shadow_to_model_q(source_frames))
        points = shadow_model.get_penetraion_keypoints().cpu().numpy()
    tip_points = {
        name: points[:, point_index, :]
        for name, point_index in SHADOW_TIP_INDICES.items()
    }
    phases = infer_motion_phases(
        source_frames,
        tip_points,
        object_vertices,
        contact_threshold,
        min_contact_tips,
        lift_delta,
        "nearest",
    )
    return points.astype(np.float32), phases


def xhand_tip_indices() -> list[int]:
    """读取XHand五个语义指尖在30点运动学输出中的索引。

    输入：无显式参数；读取冻结`xhand_keypoint_map.json`。
    输出：按食、中、环、小、拇指顺序的五个整数索引。
    内部逻辑：以semantic查找`xhand_index`并拒绝缺点。
    作用：不依赖官方候选中缺失的`mapping_semantics`元数据。
    """
    config = json.loads(
        (RETARGET_ROOT / "configs" / "xhand_keypoint_map.json").read_text(
            encoding="utf-8"
        )
    )
    by_name = {pair["semantic"]: pair for pair in config["pairs"]}
    return [int(by_name[name]["xhand_index"]) for name in TIP_SEMANTICS]


def target_tip_center(
    hand: str,
    target_model,
    data: dict,
    frames: np.ndarray,
    lift_start_frame: int,
    trajectory_position: int,
) -> np.ndarray:
    """用对应手的校准关键点计算抬升首帧五指中心。

    输入：手类型、目标模型、候选字典、单条轨迹、lift帧和候选内行号。
    输出：三维世界坐标。
    内部逻辑：Linker使用真实link局部校准点；XHand/Wuji使用各自映射JSON中的
    五个目标关键点索引，并将保存欧拉角动作转换为模型旋转6D格式。
    作用：消除三手关键点数组结构差异，为共享腕部规则提供同一物理量。
    """
    lift = int(lift_start_frame)
    if hand == "linker":
        semantics = [str(value) for value in data.get("mapping_semantics", [])]
        pairs = linker_geometry.load_pairs(semantics)
        points = linker_geometry.compute_linker_points(
            target_model,
            linker_geometry.linker_to_model_q(frames[lift : lift + 1]),
            pairs,
        )[0]
        indices = [semantics.index(name) for name in TIP_SEMANTICS]
    elif hand == "xhand":
        with torch.no_grad():
            points = (
                target_model.get_penetraion_keypoints(
                    q=xhand_geometry.xhand_to_model_q(frames[lift : lift + 1])
                )
                .cpu()
                .numpy()[0]
            )
        indices = xhand_tip_indices()
    else:
        mapping_config, semantics = trajectory_mapping_metadata(
            data,
            trajectory_position,
            RETARGET_ROOT / "configs" / "wuji_keypoint_map.json",
        )
        pairs = wuji_geometry.load_pairs(semantics, mapping_config)
        by_name = {pair["semantic"]: pair for pair in pairs}
        with torch.no_grad():
            points = (
                target_model.get_penetraion_keypoints(
                    q=wuji_geometry.wuji_to_model_q(frames[lift : lift + 1])
                )
                .cpu()
                .numpy()[0]
            )
        indices = [int(by_name[name]["wuji_index"]) for name in TIP_SEMANTICS]
    return np.mean(points[indices], axis=0).astype(np.float32)


def desired_center(
    mode: str,
    object_vertices: np.ndarray,
    shadow_points: np.ndarray,
    lift_start_frame: int,
) -> np.ndarray:
    """按实验模式计算目标手五指中心应靠近的位置。

    输入：`object_bbox`或`shadow_tips`、物体顶点、Shadow关键点和lift帧。
    输出：三维期望世界坐标。
    内部逻辑：物体模式取世界AABB极值中点；专家模式取同帧五个Shadow指尖均值。
    作用：在同一实现内公平比较“物体几何先验”和“专家抓取位置先验”。
    """
    if mode == "object_bbox":
        vertices = np.asarray(object_vertices, dtype=np.float32)
        return ((vertices.min(axis=0) + vertices.max(axis=0)) * 0.5).astype(
            np.float32
        )
    if mode == "shadow_tips":
        indices = list(SHADOW_TIP_INDICES.values())
        return np.mean(shadow_points[int(lift_start_frame), indices], axis=0).astype(
            np.float32
        )
    raise ValueError(f"未知中心目标模式: {mode}")


def existing_or_inferred_phase(data: dict, position: int, inferred: dict) -> dict:
    """优先复用已冻结Linker阶段，否则使用本轮统一推断结果。

    输入：候选字典、轨迹行号和从Shadow计算的阶段。
    输出：至少含close/lift及回退审计的阶段字典。
    内部逻辑：存在对齐`squeeze_phase_metadata`时复制该行，避免重算改变已确认Linker；
    XHand/Wuji没有该字段时复制统一推断结果。
    作用：保证共享实现对Linker物体中心模式数值兼容，同时覆盖另外两只手。
    """
    metadata = data.get("squeeze_phase_metadata")
    if isinstance(metadata, list) and len(metadata) == len(data["grasp_seqs"]):
        return dict(metadata[position])
    return {
        "close_start_frame": int(inferred["close_start_frame"]),
        "lift_start_frame": int(inferred["lift_start_frame"]),
        "grasp_frame": int(inferred["grasp_frame"]),
        "close_detection": inferred["close_detection"],
        "contact_fallback_used": bool(inferred["contact_fallback_used"]),
        "close_contact_order_distance_m": float(
            inferred["close_contact_order_distance_m"]
        ),
    }


def refine_manifest(
    hand: str,
    center_mode: str,
    manifest: dict,
    input_dir: Path,
    output_dir: Path,
    max_advance_m: float,
    source_z_offset: float,
    object_clearance: float,
    contact_threshold: float,
    min_contact_tips: int,
    lift_delta: float,
) -> dict:
    """批量生成一只手、一种中心目标和一个全局距离的候选。

    输入：手/模式、manifest、目录、距离和统一几何阶段参数。
    输出：批次摘要；同时保存每物体候选npy。
    内部逻辑：按源索引切片，逐轨迹恢复源手/目标手/物体几何，应用共享平移函数。
    作用：形成可直接用同一PhysX入口比较的单方法，而不是逐轨迹候选并集。
    """
    dimension = HAND_DIMENSIONS[hand]
    output_dir.mkdir(parents=True, exist_ok=True)
    shadow_model, target_model = build_hand_models(hand)
    records = []
    for entry in manifest.get("entries", []):
        object_name = str(entry["object_name"])
        input_path = input_dir / f"{object_name}.npy"
        full = np.load(input_path, allow_pickle=True).item()
        indices = [int(value) for value in entry["trajectory_indices"]]
        data = slice_candidate(full, indices, dimension)
        source = np.load(entry["source_path"], allow_pickle=True).item()
        frames = np.asarray(data["grasp_seqs"], dtype=np.float32)
        outputs, audits, phases = [], [], []
        for position, source_index in enumerate(indices):
            source_frames = np.asarray(
                source["grasp_seqs"][source_index], dtype=np.float32
            ).copy()
            source_frames[:, 2] += float(source_z_offset)
            vertices = transformed_object_vertices(
                Path(entry["object_asset_path"]),
                np.asarray(source["obj_scale"])[source_index],
                np.asarray(source["obj_rotmat"])[source_index],
                object_clearance,
            )
            shadow_points, inferred = shadow_points_and_phases(
                shadow_model,
                source_frames,
                vertices,
                contact_threshold,
                min_contact_tips,
                lift_delta,
            )
            phase = existing_or_inferred_phase(data, position, inferred)
            lift = int(phase["lift_start_frame"])
            grasp_center = target_tip_center(
                hand, target_model, data, frames[position], lift, position
            )
            center = desired_center(center_mode, vertices, shadow_points, lift)
            output, audit = apply_object_centric_advance(
                frames[position],
                int(phase["close_start_frame"]),
                lift,
                grasp_center,
                center,
                max_advance_m,
            )
            audit.update(
                {
                    # 底层兼容字段叫object_bbox_center，但共享方法还可以
                    # 以Shadow专家指尖为目标，因此另存一个语义准确的字段。
                    "target_center_xyz_m": center.astype(float).tolist(),
                    "source_trajectory_index": source_index,
                    "hand": hand,
                    "center_mode": center_mode,
                    "close_start_frame": int(phase["close_start_frame"]),
                    "lift_start_frame": lift,
                    "close_detection": phase.get("close_detection"),
                    "contact_fallback_used": bool(
                        phase.get("contact_fallback_used", False)
                    ),
                    "phase_source": (
                        "existing_squeeze_metadata"
                        if "squeeze_phase_metadata" in data
                        else "shadow_contact_inference"
                    ),
                }
            )
            outputs.append(output)
            audits.append(audit)
            phases.append({"source_trajectory_index": source_index, **phase})
            records.append({"object_name": object_name, **audit})
        result = dict(data)
        result.update(
            {
                "grasp_seqs": np.stack(outputs).astype(np.float32),
                "method": f"shared_grasp_center_{center_mode}_{max_advance_m * 1000:g}mm_v1",
                "shared_grasp_center_input": str(input_path.resolve()),
                "shared_grasp_center_hand": hand,
                "shared_grasp_center_mode": center_mode,
                "shared_grasp_center_max_advance_m": float(max_advance_m),
                "shared_grasp_center_audit": audits,
                "shared_grasp_center_phase_metadata": phases,
            }
        )
        np.save(output_dir / input_path.name, result, allow_pickle=True)
    expected = int(manifest.get("trajectory_count", len(records)))
    if len(records) != expected:
        raise ValueError(f"输出轨迹数{len(records)}与manifest声明{expected}不符")
    distances_before = np.asarray(
        [item["center_distance_before_m"] for item in records], dtype=np.float64
    )
    return {
        "method": f"shared_grasp_center_{center_mode}_{max_advance_m * 1000:g}mm_v1",
        "hand": hand,
        "center_mode": center_mode,
        "manifest_purpose": manifest.get("purpose"),
        "input_dir": str(input_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "max_advance_m": float(max_advance_m),
        "trajectory_count": len(records),
        "mean_center_distance_before_m": float(
            np.mean(distances_before)
        ),
        "center_distance_before_percentiles_m": {
            str(percentile): float(np.percentile(distances_before, percentile))
            for percentile in (0, 10, 25, 50, 75, 90, 100)
        },
        "mean_actual_advance_m": float(
            np.mean([item["actual_advance_m"] for item in records])
        ),
        "mean_center_distance_after_m": float(
            np.mean([item["center_distance_after_m"] for item in records])
        ),
        "contact_fallback_trajectory_count": sum(
            item["phase_source"] == "shadow_contact_inference"
            and item.get("contact_fallback_used", False)
            for item in records
        ),
        "records": records,
    }


def main() -> None:
    """解析共享校准参数，生成候选和静态摘要。

    输入：`--hand/--center-mode/--manifest/--input-dir/--output-dir`及几何阈值。
    输出：候选npy和`shared_grasp_center_summary.json`；不启动物理仿真。
    内部逻辑：毫米转米后调用批处理函数，并打印中心距离和阶段回退数。
    作用：为三手统一的少量参数筛选提供可复现入口。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand", choices=sorted(HAND_DIMENSIONS), required=True)
    parser.add_argument(
        "--center-mode", choices=("object_bbox", "shadow_tips"), required=True
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-advance-mm", type=float, required=True)
    parser.add_argument("--source-z-offset", type=float, default=0.4)
    parser.add_argument("--object-clearance", type=float, default=0.005)
    parser.add_argument("--contact-threshold", type=float, default=0.02)
    parser.add_argument("--min-contact-tips", type=int, default=2)
    parser.add_argument("--lift-delta", type=float, default=0.03)
    args = parser.parse_args()
    if args.max_advance_mm <= 0:
        parser.error("--max-advance-mm必须为正数")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    summary = refine_manifest(
        args.hand,
        args.center_mode,
        manifest,
        args.input_dir,
        args.output_dir,
        args.max_advance_mm / 1000.0,
        args.source_z_offset,
        args.object_clearance,
        args.contact_threshold,
        args.min_contact_tips,
        args.lift_delta,
    )
    summary_path = args.output_dir / "shared_grasp_center_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"hand={summary['hand']} mode={summary['center_mode']}")
    print(f"trajectories={summary['trajectory_count']}")
    print(
        "mean_center_distance_mm="
        f"{1000 * summary['mean_center_distance_before_m']:.3f}->"
        f"{1000 * summary['mean_center_distance_after_m']:.3f}"
    )
    print(f"mean_actual_advance_mm={1000 * summary['mean_actual_advance_m']:.3f}")
    print(f"contact_fallbacks={summary['contact_fallback_trajectory_count']}")
    print(f"summary={summary_path.resolve()}")


if __name__ == "__main__":
    main()
