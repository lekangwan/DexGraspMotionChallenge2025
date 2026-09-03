#!/usr/bin/env python3
"""把目标手物理专家轨迹物化为无泄漏的策略训练数组。

输入：正式manifest、对象级策略split、手类型、物理评测摘要和逐轨迹trace目录。
输出：train/valid/test三个NPZ、仅由train统计的归一化参数、对象映射和摘要JSON。
内部逻辑：核对每条trace元数据与维度，拼接特权状态观测；train/valid默认只保留
严格重放成功轨迹，test完整保留且仅附专家上限标签，绝不参与归一化统计。
作用：为单帧BC、Temporal3和Diffusion提供同一标准输入，同时阻止失败目标轨迹冒充专家。
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

try:
    from ..observations import build_object_shape_descriptor, build_observation_batch
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from observations import build_object_shape_descriptor, build_observation_batch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SPECS = (
    Path(__file__).resolve().parents[1] / "configs" / "hand_data_specs_v5.json"
)
REQUIRED_TRACE_FIELDS = {
    "hand_dof_position",
    "hand_dof_velocity",
    "policy_action",
    "object_position",
    "object_quaternion_xyzw",
    "object_linear_velocity",
    "object_angular_velocity",
    "hand_object_contact_count",
    "source_frame_index",
    "is_hold",
    "metadata_json",
}


def resolve_project_path(value):
    """把绝对路径或项目相对路径解析为绝对Path。"""
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_trace(path, hand, object_name, source_index, hand_spec):
    """读取并验证单条策略trace。

    输入：NPZ路径、预期手/物体/源索引和动作/物理维度规格。
    输出：普通数组字典及解析后的元数据。
    内部逻辑：拒绝缺字段、长度错位、维度错误或元数据错配。
    作用：防止大规模并行评测中的错文件静默污染训练集。
    """
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(REQUIRED_TRACE_FIELDS - set(archive.files))
        if missing:
            raise ValueError(f"{path}缺少trace字段: {missing}")
        arrays = {name: archive[name].copy() for name in archive.files if name != "metadata_json"}
        metadata = json.loads(str(archive["metadata_json"].item()))
    length = len(arrays["policy_action"])
    if any(len(value) != length for value in arrays.values()):
        raise ValueError(f"{path}逐步字段长度不一致")
    expected_action = int(hand_spec["policy_action_dimension"])
    expected_physics = int(hand_spec["physics_dof_dimension"])
    if arrays["policy_action"].shape != (length, expected_action):
        raise ValueError(f"{path}策略动作维度错误: {arrays['policy_action'].shape}")
    for field in ("hand_dof_position", "hand_dof_velocity"):
        if arrays[field].shape != (length, expected_physics):
            raise ValueError(f"{path}的{field}维度错误: {arrays[field].shape}")
    if (
        metadata.get("hand") != hand
        or metadata.get("object_name") != object_name
        or int(metadata.get("source_trajectory_index", -1)) != int(source_index)
    ):
        raise ValueError(f"{path}元数据与split记录不匹配")
    if metadata.get("trace_alignment") != "pre_action_state_to_command_v1":
        raise ValueError(
            f"{path}不是执行前状态到下一命令的对齐trace；禁止用于监督训练"
        )
    return arrays, metadata


def build_observations(arrays, lift_goal_m, object_shape_descriptor):
    """从物理trace构造策略状态观测。

    输入：对齐trace数组、目标抬升高度和该实例14维形状描述。
    输出：`(步骤数, O)`float32观测。
    内部逻辑：拼接手状态、物体状态、相对初始位移、剩余抬升和log接触数。
    作用：形成三手通用的特权状态BC基线，后续DexRep/点云可作为独立扩展。
    """
    return build_observation_batch(
        arrays["hand_dof_position"],
        arrays["hand_dof_velocity"],
        arrays["object_position"],
        arrays["object_quaternion_xyzw"],
        arrays["object_linear_velocity"],
        arrays["object_angular_velocity"],
        arrays["object_position"][0],
        arrays["hand_object_contact_count"],
        object_shape_descriptor,
        lift_goal_m,
    )


def save_split(path, chunks):
    """拼接并压缩保存一个策略split。

    输入：输出路径和同字段轨迹chunk列表。
    输出：该split的轨迹数、步数、观测维度和动作维度。
    内部逻辑：沿步骤维拼接数组，保留trajectory/category/object/source标签。
    作用：让训练端可随机采样，同时仍能恢复完整轨迹边界做时序窗口。
    """
    if not chunks:
        raise ValueError(f"没有可写入{path.name}的轨迹")
    fields = chunks[0].keys()
    merged = {name: np.concatenate([chunk[name] for chunk in chunks]) for name in fields}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **merged)
    return {
        "trajectory_count": len(chunks),
        "step_count": len(merged["actions"]),
        "observation_dimension": int(merged["observations"].shape[1]),
        "action_dimension": int(merged["actions"].shape[1]),
    }, merged


def compute_action_delta_limits(actions, trajectory_ids, quantile=0.995):
    """只用训练轨迹计算逐步动作变化的高分位安全范围。

    输入：未归一化动作、对应trajectory id和分位数。
    输出：逐动作维绝对变化上限与整向量L2变化上限。
    内部逻辑：只比较同一轨迹内相邻步骤，排除文件拼接边界，再取高分位并加极小下限。
    作用：闭环时限制策略产生训练专家从未出现过的剧烈位置目标跳变，而不读取valid/test。
    """
    actions = np.asarray(actions, dtype=np.float32)
    trajectory_ids = np.asarray(trajectory_ids, dtype=np.int64)
    if actions.ndim != 2 or len(actions) != len(trajectory_ids):
        raise ValueError("动作与trajectory id形状不一致")
    if not 0.0 < float(quantile) <= 1.0:
        raise ValueError("动作变化分位数必须位于(0,1]")
    same_trajectory = trajectory_ids[1:] == trajectory_ids[:-1]
    deltas = np.abs(actions[1:] - actions[:-1])[same_trajectory]
    if len(deltas) == 0:
        raise ValueError("训练数据没有同轨迹相邻动作，无法计算限速范围")
    per_dimension = np.maximum(np.quantile(deltas, quantile, axis=0), 1e-5)
    vector_norm = max(
        float(np.quantile(np.linalg.norm(deltas, axis=1), quantile)), 1e-5
    )
    return per_dimension.astype(np.float32), np.float32(vector_norm)


def prepare_dataset(args):
    """执行完整trace到策略数据集转换。

    输入：命令行参数命名空间。
    输出：摘要字典。
    内部逻辑：以split记录为唯一名单，train/valid按成功过滤，test不筛选；
    类别和物体映射按字典序冻结，归一化只读取最终train数组。
    作用：把数据质量、泄漏和归一化边界集中在一个可审计入口。
    """
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    policy_split = json.loads(args.policy_split.read_text(encoding="utf-8"))
    evaluation = json.loads(args.evaluation_summary.read_text(encoding="utf-8"))
    specs = json.loads(args.hand_specs.read_text(encoding="utf-8"))
    hand_spec = specs["hands"][args.hand]
    evaluation_manifest = Path(evaluation["manifest"]).resolve()
    same_manifest = evaluation_manifest == args.manifest.resolve()
    if not same_manifest and evaluation_manifest.is_file():
        same_manifest = (
            json.loads(evaluation_manifest.read_text(encoding="utf-8"))
            == manifest
        )
    if not same_manifest:
        raise ValueError("物理评测摘要不属于当前正式manifest")
    if evaluation.get("hand") != args.hand:
        raise ValueError("物理评测摘要的hand与请求不一致")

    manifest_entries = {entry["object_name"]: entry for entry in manifest["entries"]}
    result_by_key = {
        (item["object_name"], int(item["source_trajectory_index"])): item
        for item in evaluation["results"]
    }
    categories = sorted({record["category"] for record in policy_split["records"]})
    objects = sorted({record["object_name"] for record in policy_split["records"]})
    category_to_id = {name: index for index, name in enumerate(categories)}
    object_to_id = {name: index for index, name in enumerate(objects)}
    chunks = defaultdict(list)
    skipped = Counter()
    included_by_category = defaultdict(Counter)
    trajectory_id = 0
    policy_action_order = None
    source_cache = {}
    shape_cache = {}

    for record in policy_split["records"]:
        split = record["split"]
        object_name = record["object_name"]
        source_index = int(record["source_trajectory_index"])
        if object_name not in manifest_entries:
            raise ValueError(f"策略split含manifest外物体: {object_name}")
        result = result_by_key.get((object_name, source_index))
        if result is None:
            raise ValueError(f"评测缺少{object_name}:{source_index}")
        if (
            split in {"train", "valid"}
            and not bool(getattr(args, "include_all_train_valid", False))
            and not bool(result["success"])
        ):
            skipped[f"{split}_failed_replay"] += 1
            continue
        trace_path = args.trace_dir / object_name / f"source_{source_index}_trace.npz"
        arrays, metadata = load_trace(
            trace_path, args.hand, object_name, source_index, hand_spec
        )
        trace_action_order = list(metadata.get("policy_action_order", []))
        if len(trace_action_order) != int(hand_spec["policy_action_dimension"]):
            raise ValueError(f"{trace_path}缺少完整policy_action_order")
        if policy_action_order is None:
            policy_action_order = trace_action_order
        elif trace_action_order != policy_action_order:
            raise ValueError(f"{trace_path}的动作顺序与同手其他trace不一致")
        cache_key = (object_name, source_index)
        if cache_key not in shape_cache:
            entry = manifest_entries[object_name]
            if object_name not in source_cache:
                source_cache[object_name] = np.load(
                    entry["source_path"], allow_pickle=True
                ).item()
            scale = float(np.asarray(source_cache[object_name]["obj_scale"])[source_index])
            mesh_path = Path(entry["object_asset_path"]) / "coacd" / "decomposed.obj"
            shape_cache[cache_key] = build_object_shape_descriptor(mesh_path, scale)
        observations = build_observations(
            arrays, args.lift_goal, shape_cache[cache_key]
        )
        step_count = len(observations)
        chunk = {
            "observations": observations,
            "actions": arrays["policy_action"].astype(np.float32),
            "trajectory_id": np.full(step_count, trajectory_id, dtype=np.int64),
            "category_id": np.full(
                step_count, category_to_id[record["category"]], dtype=np.int64
            ),
            "object_id": np.full(
                step_count, object_to_id[object_name], dtype=np.int64
            ),
            "source_trajectory_index": np.full(
                step_count, source_index, dtype=np.int64
            ),
            "source_frame_index": arrays["source_frame_index"].astype(np.int16),
            "is_hold": arrays["is_hold"].astype(bool),
            "expert_replay_success": np.full(
                step_count, bool(result["success"]), dtype=bool
            ),
        }
        chunks[split].append(chunk)
        included_by_category[record["category"]][split] += 1
        trajectory_id += 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_summaries = {}
    merged_train = None
    for split in ("train", "valid", "test"):
        info, merged = save_split(args.output_dir / f"{split}.npz", chunks[split])
        split_summaries[split] = info
        if split == "train":
            merged_train = merged
    observation_mean = merged_train["observations"].mean(axis=0)
    observation_std = np.maximum(merged_train["observations"].std(axis=0), 1e-6)
    action_mean = merged_train["actions"].mean(axis=0)
    action_std = np.maximum(merged_train["actions"].std(axis=0), 1e-6)
    delta_quantile = 0.995
    action_delta_limit, action_delta_norm_limit = compute_action_delta_limits(
        merged_train["actions"], merged_train["trajectory_id"], delta_quantile
    )
    np.savez_compressed(
        args.output_dir / "normalization.npz",
        observation_mean=observation_mean.astype(np.float32),
        observation_std=observation_std.astype(np.float32),
        action_mean=action_mean.astype(np.float32),
        action_std=action_std.astype(np.float32),
        action_delta_limit=action_delta_limit,
        action_delta_norm_limit=np.asarray(action_delta_norm_limit, dtype=np.float32),
        action_delta_quantile=np.asarray(delta_quantile, dtype=np.float32),
    )
    mappings = {
        "category_to_id": category_to_id,
        "object_to_id": object_to_id,
        "policy_action_order": policy_action_order,
    }
    (args.output_dir / "mappings.json").write_text(
        json.dumps(mappings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    missing_train_categories = sorted(
        category for category in categories if included_by_category[category]["train"] == 0
    )
    summary = {
        "schema_version": 2,
        "status": "ready" if not missing_train_categories else "ready_with_gaps",
        "hand": args.hand,
        "manifest": str(args.manifest.resolve()),
        "policy_split": str(args.policy_split.resolve()),
        "evaluation_summary": str(args.evaluation_summary.resolve()),
        "trace_dir": str(args.trace_dir.resolve()),
        "quality_rule": (
            "train_valid_unfiltered_for_downstream_v3_gate; test_unfiltered"
            if bool(getattr(args, "include_all_train_valid", False))
            else "train_and_valid_strict_replay_success_only; test_unfiltered"
        ),
        "normalization_rule": "train_steps_only",
        "lift_goal_m": float(args.lift_goal),
        "runtime_action_rate_limit": {
            "source": "train_same_trajectory_adjacent_action_delta",
            "quantile": delta_quantile,
            "per_dimension_limit": action_delta_limit.tolist(),
            "vector_l2_limit": float(action_delta_norm_limit),
        },
        "split_summaries": split_summaries,
        "skipped": dict(skipped),
        "missing_successful_train_categories": missing_train_categories,
        "category_count": len(categories),
        "observation_dimension": split_summaries["train"]["observation_dimension"],
        "action_dimension": split_summaries["train"]["action_dimension"],
    }
    (args.output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main():
    """解析数据路径，物化三split并打印就绪状态。

    输入：manifest/split/hand/trace/评测摘要/输出目录。
    输出：NPZ数据集、统计与`POLICY_DATASET`终端标志。
    内部逻辑：规范化全部路径后调用`prepare_dataset`，可选严格要求类别全覆盖。
    作用：作为大规模策略训练前唯一正式数据准备入口。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--policy-split", type=Path, required=True)
    parser.add_argument("--hand", choices=["linker", "xhand", "wuji"], required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--evaluation-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hand-specs", type=Path, default=DEFAULT_SPECS)
    parser.add_argument("--lift-goal", type=float, default=0.30)
    parser.add_argument("--require-all-train-categories", action="store_true")
    parser.add_argument(
        "--include-all-train-valid",
        action="store_true",
        help="不在本步按旧result.success筛选；供v3审计sidecar随后统一过滤",
    )
    args = parser.parse_args()
    for name in ("manifest", "policy_split", "trace_dir", "evaluation_summary", "output_dir", "hand_specs"):
        setattr(args, name, resolve_project_path(getattr(args, name)))
    summary = prepare_dataset(args)
    if args.require_all_train_categories and summary["missing_successful_train_categories"]:
        raise RuntimeError(
            "以下类别没有成功训练专家轨迹: "
            + ", ".join(summary["missing_successful_train_categories"])
        )
    print(f"split_summaries={summary['split_summaries']}")
    print(f"missing_train_categories={len(summary['missing_successful_train_categories'])}")
    print(f"POLICY_DATASET={summary['status'].upper()}")


if __name__ == "__main__":
    main()
