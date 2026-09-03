#!/usr/bin/env python3
"""在物理成功门之上评估重定向专家轨迹是否平滑、可跟踪且形成多指抓握。"""

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
FINAL = ROOT / "outputs/reboot_synergy_rank5_formal1000_v1/postconfirmed_rank5_v1"
OUTPUT = Path(__file__).resolve().parents[1] / "data/final/EXPERT_LEARNABILITY_AUDIT.json"


def finger_family(hand, body):
    """把不同URDF的接触link名称统一为thumb/index/middle/ring/little/palm。"""
    name = body.lower()
    if hand == "wuji":
        for index, family in enumerate(("thumb", "index", "middle", "ring", "little"), 1):
            if f"finger{index}_" in name:
                return family
    aliases = {
        "thumb": ("thumb",), "index": ("index",), "middle": ("middle",),
        "ring": ("ring",), "little": ("pinky", "little"),
    }
    for family, words in aliases.items():
        if any(word in name for word in words):
            return family
    return "palm"


def percentile(values, prefer_high):
    """把同手不同量纲指标转成0到1的相对质量分。"""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(np.argsort(values, kind="stable"), kind="stable")
    score = order / max(len(values) - 1, 1)
    return score if prefer_high else 1.0 - score


def audit_hand(hand):
    """读取一只手的稳定审计和物理报告，输出逐轨迹可学习性指标。"""
    source = json.loads((FINAL / f"audit/{hand}_stable_audit.json").read_text(encoding="utf-8"))
    rows = []
    for item in source["results"]:
        if not item.get("training_eligible", False):
            continue
        physics = json.loads(Path(item["physics_report"]).read_text(encoding="utf-8"))
        contacts = physics.get("hand_body_contact_steps", {})
        families = {
            finger_family(hand, body)
            for body, steps in contacts.items() if int(steps) > 0
        }
        non_thumb = families & {"index", "middle", "ring", "little"}
        counts = np.asarray(physics["hand_object_contact_count_per_step"])
        first_contact = int(np.argmax(counts > 0)) if np.any(counts > 0) else len(counts)
        rows.append({
            "object_name": item["object_name"],
            "category": item["category"],
            "source_trajectory_index": int(item["source_trajectory_index"]),
            "policy_split": item.get("policy_split"),
            "keypoint_mean_distance_m": float(item["keypoint_mean_distance_m"]),
            "max_joint_step_l2_rad": float(item["max_joint_step_l2_rad"]),
            "worst_tracking_error": float(physics.get("worst_absolute_tracking_error", 0.0)),
            "peak_to_final_drop_m": float(item["peak_to_final_drop_m"]),
            "terminal_lift_range_m": float(item["terminal_lift_range_m"]),
            "transport_translation_change_m": float(item["max_palm_relative_translation_change_m"]),
            "transport_rotation_change_deg": float(item["max_palm_relative_rotation_change_deg"]),
            "terminal_contact_ratio": float(item["terminal_contact_ratio"]),
            "contact_finger_count": len(families - {"palm"}),
            "thumb_contact": "thumb" in families,
            "opposition_contact": "thumb" in families and len(non_thumb) >= 2,
            "first_contact_step": first_contact,
        })
    metrics = [
        ("keypoint_mean_distance_m", False), ("max_joint_step_l2_rad", False),
        ("worst_tracking_error", False), ("peak_to_final_drop_m", False),
        ("terminal_lift_range_m", False), ("transport_translation_change_m", False),
        ("transport_rotation_change_deg", False), ("contact_finger_count", True),
    ]
    score_parts = []
    for name, prefer_high in metrics:
        values = [row[name] for row in rows]
        scores = percentile(values, prefer_high)
        score_parts.append(scores)
        for row, score in zip(rows, scores):
            row[f"score_{name}"] = float(score)
    total = np.mean(np.stack(score_parts), axis=0)
    total = 0.85 * total + 0.15 * np.asarray([row["opposition_contact"] for row in rows])
    threshold = float(np.median(total))
    for row, score in zip(rows, total):
        row["learnability_score"] = float(score)
        row["quality_tier"] = "A" if score >= threshold else "B"
    tier_a = [row for row in rows if row["quality_tier"] == "A"]
    return {
        "eligible_count": len(rows), "tier_a_count": len(tier_a),
        "tier_a_threshold": threshold,
        "tier_a_category_count": len({row["category"] for row in tier_a}),
        "opposition_contact_count": sum(row["opposition_contact"] for row in rows),
        "metric_medians": {
            name: float(np.median([row[name] for row in rows]))
            for name, _ in metrics
        },
        "results": rows,
    }


def main():
    output = {
        "schema_version": 1,
        "meaning": "A/B仅衡量已通过物理成功门的示范可学习性；不改变最终成功率定义",
        "hands": {hand: audit_hand(hand) for hand in ("linker", "xhand", "wuji")},
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for hand, summary in output["hands"].items():
        print(hand, {key: value for key, value in summary.items() if key != "results"})
    print(OUTPUT)


if __name__ == "__main__":
    main()
