"""测试官方类别到inventory以及正式manifest硬门的纯本地逻辑。"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


RETARGET_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RETARGET_ROOT / "scripts"))

from build_inventory import inspect_object, load_category_map, write_inventory
from build_embedded_category_map import collect_embedded_records, parse_embedded_category
from freeze_formal_experiment import sha256, verify_manifest
from verify_formal_bundle import verify_lock
from select_report_cases import choose_cases
from render_selected_cases import reusable_video
from render_software_replay import forward_link_positions, parse_urdf_tree, HAND_URDFS
from export_result_tables import markdown_table, summary_row

sys.path.insert(0, str(RETARGET_ROOT / "advanced_policy" / "prepare"))
from select_expert_video_audit import build_selections, trajectory_key


def make_object_fixture(root, object_id, trajectory_count=2):
    """创建一份标准轨迹npy和两个最小COACD占位文件。"""
    trajectory_root = root / "trajectories"
    asset_root = root / "assets"
    trajectory_root.mkdir(parents=True, exist_ok=True)
    coacd = asset_root / object_id / "coacd"
    coacd.mkdir(parents=True, exist_ok=True)
    (coacd / "coacd_1.urdf").write_text("<robot name='fixture'/>", encoding="utf-8")
    (coacd / "decomposed.obj").write_text("v 0 0 0\n", encoding="utf-8")
    np.save(
        trajectory_root / f"{object_id}.npy",
        {
            "grasp_seqs": np.zeros((trajectory_count, 70, 28), dtype=np.float32),
            "obj_scale": np.ones(trajectory_count, dtype=np.float32),
            "obj_rotmat": np.tile(np.eye(3), (trajectory_count, 1, 1)),
        },
        allow_pickle=True,
    )
    return trajectory_root, asset_root


class FormalToolsTest(unittest.TestCase):
    """验证正式数据入口不会接受错配或不完整数量。"""

    def test_category_map_builds_exact_inventory_row(self):
        """类别表中的物体应匹配同名轨迹/资产并写出真实轨迹数。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trajectory_root, asset_root = make_object_fixture(root, "cup_a", 3)
            category_map = root / "category.csv"
            category_map.write_text("object_id,category\ncup_a,cup\n", encoding="utf-8")
            records = load_category_map(category_map)
            row = inspect_object(records[0], trajectory_root, asset_root)
            self.assertEqual(row["trajectory_count"], 3)
            inventory = root / "inventory.csv"
            write_inventory(inventory, [row])
            with inventory.open(newline="", encoding="utf-8") as handle:
                loaded = list(csv.DictReader(handle))
            self.assertEqual(loaded[0]["object_id"], "cup_a")
            self.assertEqual(loaded[0]["category"], "cup")

    def test_embedded_categories_merge_sources_and_reject_ambiguous_ids(self):
        """core/sem同名类别应合并，mujoco商品名不能被猜成类别。"""
        self.assertEqual(parse_embedded_category("core-bottle-a"), ("core", "bottle"))
        self.assertEqual(parse_embedded_category("sem-Bottle-b"), ("sem", "bottle"))
        self.assertIsNone(parse_embedded_category("mujoco-Some_Product"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trajectory_root = root / "trajectories"
            asset_root = root / "assets"
            for object_id in ("core-bottle-a", "sem-Bottle-b", "mujoco-Some_Product"):
                make_object_fixture(root, object_id, 10)
                source = root / "trajectories" / f"{object_id}.npy"
                target = trajectory_root / source.name
                self.assertEqual(source, target)
            records, audit = collect_embedded_records(trajectory_root, asset_root, 10, 2)
            self.assertEqual([item["category"] for item in records], ["bottle", "bottle"])
            self.assertEqual(audit["eligible_category_count"], 1)
            self.assertEqual(audit["excluded_counts"]["unsupported_object_family"], 1)

    def test_manifest_gate_checks_partition_and_source_hash(self):
        """小型1类2物体协议应通过，篡改源文件后必须被哈希门拒绝。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entries = []
            for object_id in ("cup_a", "cup_b"):
                trajectory_root, asset_root = make_object_fixture(root / object_id, object_id, 2)
                source = trajectory_root / f"{object_id}.npy"
                entries.append(
                    {
                        "object_name": object_id,
                        "category": "cup",
                        "source_path": str(source),
                        "source_sha256": sha256(source),
                        "object_asset_path": str(asset_root / object_id),
                        "available_trajectory_count": 2,
                        "trajectory_indices": [0, 1],
                        "calibration_indices": [0],
                        "heldout_indices": [1],
                    }
                )
            manifest = {"selection_seed": 7, "entries": entries}
            protocol = {
                "selection_seed": 7, "category_count": 1, "object_count": 2,
                "trajectory_count": 4, "objects_per_category": 2,
                "trajectories_per_object": 2, "calibration_per_object": 1,
                "heldout_per_object": 1,
            }
            self.assertEqual(len(verify_manifest(manifest, protocol, True)), 2)
            Path(entries[0]["source_path"]).write_bytes(b"changed")
            with self.assertRaises(ValueError):
                verify_manifest(manifest, protocol, True)

    def test_lock_detects_category_input_drift(self):
        """冻结后的类别或inventory凭据发生变化时，输入门必须拒绝。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment = root / "experiment.json"
            manifest = root / "manifest.json"
            category_input = root / "category.csv"
            implementation = root / "implementation.py"
            experiment.write_text("{}\n", encoding="utf-8")
            manifest.write_text("{}\n", encoding="utf-8")
            category_input.write_text("object_id,category\n", encoding="utf-8")
            implementation.write_text("VALUE = 1\n", encoding="utf-8")
            relative_input = os.path.relpath(category_input, RETARGET_ROOT.parent)
            relative_implementation = os.path.relpath(implementation, RETARGET_ROOT.parent)
            lock = {
                "experiment_config_sha256": sha256(experiment),
                "manifest_sha256": sha256(manifest),
                "input_fingerprints": {relative_input: sha256(category_input)},
                "implementation_fingerprints": {
                    relative_implementation: sha256(implementation)
                },
            }
            lock_path = root / "lock.json"
            lock_path.write_text(json.dumps(lock), encoding="utf-8")
            result = verify_lock(lock_path, experiment, manifest)
            self.assertEqual(result["input_fingerprint_count"], 1)
            category_input.write_text("object_id,category\na,cup\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "输入凭据变化"):
                verify_lock(lock_path, experiment, manifest)

    def test_report_case_selection_is_unique_and_finds_slip(self):
        """同一轨迹不能进入多个视频组，越过10cm后掉落应进入滑落组。"""
        results = [
            {
                "object_name": "a", "category": "cup", "source_trajectory_index": 0,
                "success": True, "max_lift_m": 0.2, "final_lift_m": 0.19,
                "longest_sustained_lift_time_s": 1.0, "hand_object_contact_steps": 100,
            },
            {
                "object_name": "b", "category": "bottle", "source_trajectory_index": 1,
                "success": False, "max_lift_m": 0.13, "final_lift_m": 0.03,
                "longest_sustained_lift_time_s": 0.2, "hand_object_contact_steps": 80,
            },
            {
                "object_name": "c", "category": "bowl", "source_trajectory_index": 2,
                "success": False, "max_lift_m": 0.09, "final_lift_m": 0.08,
                "longest_sustained_lift_time_s": 0.4, "hand_object_contact_steps": 40,
            },
            {
                "object_name": "d", "category": "camera", "source_trajectory_index": 3,
                "success": False, "max_lift_m": 0.01, "final_lift_m": 0.0,
                "longest_sustained_lift_time_s": 0.0, "hand_object_contact_steps": 20,
            },
        ]
        groups = choose_cases(results, 1)
        self.assertEqual(groups["lift_then_slip"][0]["object_name"], "b")
        keys = [
            (item["object_name"], item["source_trajectory_index"])
            for items in groups.values() for item in items
        ]
        self.assertEqual(len(keys), len(set(keys)))

    def test_expert_video_audit_is_train_only_and_cross_hand_aligned(self):
        """成功视频审查应每手20条，前10条三手同轨迹且不泄漏test。"""
        def result(hand, object_name, category, index, quality):
            """生成具有排序指标的最小成功轨迹记录。"""
            return {
                "hand": hand,
                "object_name": object_name,
                "category": category,
                "source_trajectory_index": index,
                "target_trajectory_index": index,
                "success": True,
                "max_lift_m": 0.11 + quality * 0.001,
                "final_lift_m": quality * 0.001,
                "max_xy_drift_m": 0.0,
                "longest_sustained_lift_time_s": 0.5 + quality * 0.01,
                "hand_object_contact_steps": 30 + quality,
                "max_simultaneous_hand_object_contacts": 2,
                "physics_report": f"/{hand}/{object_name}.json",
            }

        records = []
        reports = {hand: {"results": []} for hand in ("linker", "xhand", "wuji")}
        for index in range(12):
            object_name = f"shared_{index}"
            records.append(
                {"object_name": object_name, "source_trajectory_index": index, "split": "train"}
            )
            for hand in reports:
                reports[hand]["results"].append(
                    result(hand, object_name, f"shared_category_{index}", index, index)
                )
        for hand_index, hand in enumerate(reports):
            for index in range(12):
                object_name = f"{hand}_only_{index}"
                records.append(
                    {"object_name": object_name, "source_trajectory_index": index, "split": "train"}
                )
                reports[hand]["results"].append(
                    result(hand, object_name, f"{hand}_category_{index}", index, index)
                )
            test_name = f"{hand}_test_success"
            records.append(
                {"object_name": test_name, "source_trajectory_index": 99, "split": "test"}
            )
            reports[hand]["results"].append(
                result(hand, test_name, f"test_category_{hand_index}", 99, 99)
            )

        selections, common_keys, stats = build_selections({"records": records}, reports)
        self.assertEqual(stats["common_pool_count"], 12)
        self.assertEqual(len(common_keys), 10)
        expected_shared = None
        for hand, selection in selections.items():
            self.assertEqual(sum(len(items) for items in selection["groups"].values()), 20)
            shared = [trajectory_key(item) for item in selection["groups"]["shared_success"]]
            self.assertEqual(shared, common_keys)
            self.assertFalse(any("test_success" in item["object_name"] for items in selection["groups"].values() for item in items))
            expected_shared = shared if expected_shared is None else expected_shared
            self.assertEqual(shared, expected_shared, hand)

    def test_isaac_video_resume_requires_matching_complete_report(self):
        """Isaac续跑只能复用非空MP4与完整匹配的同名JSON。"""
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "case.mp4"
            video.write_bytes(b"0" * 2048)
            item = {"object_name": "cup", "source_trajectory_index": 7, "success": True}
            report = {
                "object_name": "cup",
                "source_trajectory_index": 7,
                "success": True,
                "video_frame_count": 80,
                "video": str(video.resolve()),
            }
            video.with_suffix(".json").write_text(json.dumps(report), encoding="utf-8")
            self.assertTrue(reusable_video(video, item, "isaac"))
            report["source_trajectory_index"] = 8
            video.with_suffix(".json").write_text(json.dumps(report), encoding="utf-8")
            self.assertFalse(reusable_video(video, item, "isaac"))

    def test_software_renderer_forward_kinematics_is_finite(self):
        """三只手6D URDF的零位骨架都应能在纯CPU前向运动学中完整展开。"""
        for hand, urdf in HAND_URDFS.items():
            root, by_parent, edges = parse_urdf_tree(urdf)
            positions = forward_link_positions(root, by_parent, {})
            self.assertGreater(len(positions), 10, hand)
            self.assertEqual(len(edges), len(positions) - 1, hand)
            self.assertTrue(np.isfinite(np.stack(list(positions.values()))).all(), hand)

    def test_result_table_preserves_micro_and_macro_rates(self):
        """导出表必须分别保留轨迹微平均、物体宏平均和类别宏平均。"""
        summary = {
            "hand": "xhand", "trajectory_count": 10, "success_count": 4,
            "trajectory_micro_success_rate": 0.4,
            "object_macro_success_rate": 0.35,
            "category_macro_success_rate": 0.3,
            "mean_max_lift_m": 0.12,
            "results": [],
        }
        row = summary_row("method", Path("/tmp/fake.json"), summary)
        self.assertEqual(row["trajectory_micro_success_rate"], 0.4)
        self.assertEqual(row["object_macro_success_rate"], 0.35)
        self.assertEqual(row["category_macro_success_rate"], 0.3)
        table = markdown_table([row])
        self.assertIn("40.00%", table)
        self.assertIn("35.00%", table)
        self.assertIn("30.00%", table)


if __name__ == "__main__":
    unittest.main()
