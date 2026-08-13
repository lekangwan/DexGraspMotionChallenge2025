#!/usr/bin/env python3
"""从多个Wuji重定向候选中逐轨迹选择物理表现最佳者。

输入：冻结manifest、若干`名称 候选目录 评估目录`三元组和输出目录。
输出：每物体混合候选npy及逐轨迹选择理由、候选分数和预计成功率JSON。
内部逻辑：成功优先，再比较持续抬升、最终/最大高度和接触步数；不重新优化动作。
作用：实现“多种几何假设提出抓法，物理重放负责选择”的仿真闭环重定向方法。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from wuji_candidate_utils import (
    physics_selection_score,
    trajectory_mapping_metadata,
)


EVALUATE_DIR = Path(__file__).resolve().parent
DEFAULT_MAPPING = EVALUATE_DIR.parent / "configs" / "wuji_keypoint_map.json"


def load_candidate(name, target_dir, evaluation_dir):
    """加载一个候选方法的评估摘要并建立轨迹索引。

    输入：方法名、候选目录和统一评估目录。
    输出：包含路径、原摘要和`(物体,源索引)`结果字典的结构。
    内部逻辑：读取统一评估摘要并拒绝重复轨迹键。
    作用：把命令行三元组转换为后续选择可快速查询的数据源。
    """
    target_dir = Path(target_dir).resolve()
    evaluation_dir = Path(evaluation_dir).resolve()
    summary_path = evaluation_dir / "manifest_evaluation_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("hand") != "wuji":
        raise ValueError(f"{name}评估摘要不是Wuji: {summary_path}")
    by_key = {}
    for result in summary["results"]:
        key = (result["object_name"], int(result["source_trajectory_index"]))
        if key in by_key:
            raise ValueError(f"{name}包含重复轨迹: {key}")
        by_key[key] = result
    return {
        "name": name,
        "target_dir": target_dir,
        "evaluation_dir": evaluation_dir,
        "summary": summary,
        "by_key": by_key,
    }


def validate_candidate_file(candidate, entry):
    """验证某方法的单物体候选与manifest完全对齐。

    输入：候选方法结构和manifest物体条目。
    输出：加载后的npy字典。
    内部逻辑：检查文件、`(N,70,26)`形状、源索引以及评估摘要键。
    作用：防止从错目录或不完整评估中拼接轨迹。
    """
    path = candidate["target_dir"] / f"{entry['object_name']}.npy"
    if not path.is_file():
        raise FileNotFoundError(f"候选文件不存在: {path}")
    data = np.load(path, allow_pickle=True).item()
    indices = [int(value) for value in entry["trajectory_indices"]]
    if not np.array_equal(data["source_trajectory_indices"], indices):
        raise ValueError(f"候选索引不匹配: {path}")
    if np.asarray(data["grasp_seqs"]).shape != (len(indices), 70, 26):
        raise ValueError(f"候选形状不匹配: {path}")
    missing = [
        index
        for index in indices
        if (entry["object_name"], index) not in candidate["by_key"]
    ]
    if missing:
        raise ValueError(f"评估摘要缺少轨迹: {path} {missing}")
    return data


def select_object(entry, candidates, output_dir):
    """为一个物体逐轨迹选择并保存最佳Wuji候选。

    输入：manifest条目、全部候选方法和输出目录。
    输出：逐轨迹选择记录列表，同时写出混合npy。
    内部逻辑：对同一源索引计算物理分数，复制最高分动作及其映射元数据。
    作用：保留v1/v2互补成功轨迹，而不是为所有物体固定一种映射。
    """
    loaded = {
        candidate["name"]: validate_candidate_file(candidate, entry)
        for candidate in candidates
    }
    indices = [int(value) for value in entry["trajectory_indices"]]
    selected_frames = []
    selected_configs = []
    selected_semantics = []
    selections = []
    for target_index, source_index in enumerate(indices):
        evaluated = []
        for candidate in candidates:
            metrics = candidate["by_key"][(entry["object_name"], source_index)]
            evaluated.append((physics_selection_score(metrics), candidate, metrics))
        score, selected, metrics = max(evaluated, key=lambda item: item[0])
        selected_data = loaded[selected["name"]]
        mapping, semantics = trajectory_mapping_metadata(
            selected_data, target_index, DEFAULT_MAPPING
        )
        selected_frames.append(selected_data["grasp_seqs"][target_index])
        selected_configs.append(str(mapping.resolve()))
        selected_semantics.append(semantics)
        selections.append(
            {
                "object_name": entry["object_name"],
                "source_trajectory_index": source_index,
                "target_trajectory_index": target_index,
                "selected_candidate": selected["name"],
                "selected_score": list(score),
                "selected_success": bool(metrics["success"]),
                "candidate_metrics": {
                    item_candidate["name"]: {
                        "score": list(item_score),
                        "success": bool(item_metrics["success"]),
                        "max_lift_m": item_metrics["max_lift_m"],
                        "final_lift_m": item_metrics["final_lift_m"],
                        "longest_sustained_lift_time_s": item_metrics[
                            "longest_sustained_lift_time_s"
                        ],
                        "hand_object_contact_steps": item_metrics[
                            "hand_object_contact_steps"
                        ],
                    }
                    for item_score, item_candidate, item_metrics in evaluated
                },
            }
        )
    first = loaded[candidates[0]["name"]]
    joint_names = list(first["wuji_joint_names"])
    for candidate in candidates[1:]:
        if list(loaded[candidate["name"]]["wuji_joint_names"]) != joint_names:
            raise ValueError("候选方法的Wuji关节顺序不一致")
    output = {
        "grasp_seqs": np.stack(selected_frames).astype(np.float32),
        "source_trajectory_indices": np.asarray(indices, dtype=np.int64),
        "obj_rotmat": np.asarray(first["obj_rotmat"]),
        "obj_scale": np.asarray(first["obj_scale"]),
        "wuji_joint_names": joint_names,
        "source_z_offset": float(first.get("source_z_offset", 0.4)),
        "mapping_config_per_trajectory": selected_configs,
        "mapping_semantics_per_trajectory": selected_semantics,
        "selected_candidate_per_trajectory": [
            selection["selected_candidate"] for selection in selections
        ],
        "selection_method": (
            "lexicographic(success,sustained_time,final_lift,max_lift,contact_steps)"
        ),
    }
    output_path = output_dir / f"{entry['object_name']}.npy"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, output, allow_pickle=True)
    return selections


def main():
    """解析候选三元组、执行逐轨迹选择并保存总摘要。

    输入：manifest、重复的`--candidate NAME TARGET_DIR EVAL_DIR`和输出目录。
    输出：混合候选文件及`candidate_selection_summary.json`。
    内部逻辑：要求至少两个候选，逐物体选择后汇总各方法入选数和预计成功率。
    作用：作为物理在环多假设重定向的标准离线选择入口。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidate", nargs=3, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if len(args.candidate) < 2:
        raise ValueError("多候选选择至少需要两种方法")
    names = [values[0] for values in args.candidate]
    if len(names) != len(set(names)):
        raise ValueError("候选方法名不能重复")
    candidates = [load_candidate(*values) for values in args.candidate]
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selections = []
    for entry in manifest["entries"]:
        selections.extend(select_object(entry, candidates, args.output_dir))
    counts = {
        name: sum(item["selected_candidate"] == name for item in selections)
        for name in names
    }
    success_count = sum(item["selected_success"] for item in selections)
    summary = {
        "hand": "wuji",
        "manifest": str(args.manifest.resolve()),
        "candidate_names": names,
        "selection_method": (
            "lexicographic(success,sustained_time,final_lift,max_lift,contact_steps)"
        ),
        "trajectory_count": len(selections),
        "predicted_success_count": success_count,
        "predicted_success_rate": success_count / len(selections),
        "selected_count_by_candidate": counts,
        "selections": selections,
    }
    summary_path = args.output_dir / "candidate_selection_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"selected_count={counts}")
    print(f"predicted_success={success_count}/{len(selections)}")
    print(f"output={summary_path}")


if __name__ == "__main__":
    main()
