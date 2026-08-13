#!/usr/bin/env python3
"""审计三只目标手的URDF自由度与关键点文件。

本脚本不依赖Isaac Gym或GPU，适合在正式优化前运行。它只回答：
1. URDF中有多少非固定、mimic和主动关节；
2. 关键点JSON覆盖多少link；
3. 有多少关键点仍是局部坐标原点占位符。

局部坐标为零不一定永远错误（link原点有时就是合理标记），但如果整只手全部
为零，就必须经过可视化校准，不能直接把优化损失当作可信结果。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "reference" / "HandRetargetTask2026" / "scripts" / "assets"

HANDS = {
    "xhand": {
        "expected_active": 12,
        "urdf": ASSETS / "xhand_right" / "urdf" / "xhand_right.urdf",
        "keypoints": ASSETS / "xhand_right" / "urdf" / "penetration_xhand.json",
    },
    "linker_o6": {
        "expected_active": 6,
        "urdf": ASSETS / "linkerhand" / "o6" / "right" / "linkerhand_o6_right.urdf",
        "keypoints": ASSETS / "linkerhand" / "o6" / "right" / "penetration_linkerhand.json",
    },
    "wuji": {
        "expected_active": 20,
        "urdf": ASSETS / "wujihand_urdf" / "urdf" / "right.urdf",
        "keypoints": ASSETS / "wujihand_urdf" / "urdf" / "penetration_wuji_right.json",
    },
}


def audit_urdf(path: Path) -> dict:
    root = ET.parse(path).getroot()
    movable = [joint for joint in root.findall("joint") if joint.get("type") != "fixed"]
    mimic = [joint for joint in movable if joint.find("mimic") is not None]
    active = [joint for joint in movable if joint.find("mimic") is None]
    limits_missing = [
        joint.get("name")
        for joint in active
        if joint.get("type") in {"revolute", "prismatic"}
        and joint.find("limit") is None
    ]
    return {
        "movable_joint_count": len(movable),
        "mimic_joint_count": len(mimic),
        "active_joint_count": len(active),
        "active_joint_names": [joint.get("name") for joint in active],
        "mimic_joint_names": [joint.get("name") for joint in mimic],
        "active_limits_missing": limits_missing,
    }


def audit_keypoints(path: Path) -> dict:
    values = json.loads(path.read_text(encoding="utf-8"))
    point_count = 0
    origin_count = 0
    malformed_links = []
    for link, flat_points in values.items():
        if not isinstance(flat_points, list) or len(flat_points) % 3:
            malformed_links.append(link)
            continue
        for start in range(0, len(flat_points), 3):
            point = flat_points[start : start + 3]
            point_count += 1
            if all(abs(float(value)) <= 1e-12 for value in point):
                origin_count += 1
    return {
        "link_count": len(values),
        "point_count": point_count,
        "origin_placeholder_count": origin_count,
        "nonzero_local_point_count": point_count - origin_count,
        "malformed_links": malformed_links,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    report = {}
    failed = False
    for name, config in HANDS.items():
        urdf = audit_urdf(config["urdf"])
        keypoints = audit_keypoints(config["keypoints"])
        expected = config["expected_active"]
        active_ok = urdf["active_joint_count"] == expected
        failed |= not active_ok or bool(urdf["active_limits_missing"])
        report[name] = {
            "expected_active": expected,
            "active_count_ok": active_ok,
            "urdf": urdf,
            "keypoints": keypoints,
        }

        print(
            f"{name}: active={urdf['active_joint_count']}/{expected}, "
            f"mimic={urdf['mimic_joint_count']}, "
            f"keypoints={keypoints['point_count']}, "
            f"local_origin={keypoints['origin_placeholder_count']}"
        )

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"report={args.json_out}")

    print("HAND_ASSET_AUDIT=" + ("FAIL" if failed else "PASS"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

