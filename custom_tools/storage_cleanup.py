"""Safely remove unselected training weights while preserving final evidence.

The default mode is preview-only.  Destructive cleanup requires both
``--execute`` and the exact confirmation phrase printed by the preview.
Experiment configs, logs, TensorBoard files, evaluation YAML/CSV files, and
source code are never selected for deletion.
"""

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BC_ROOT = ROOT / "custom_tools/runs/bc"
RESIDUAL_ROOT = ROOT / "custom_tools/runs/residual_ppo"
DEFAULT_MANIFEST = (
    ROOT / "custom_tools/results/storage_cleanup_manifest_v1.json")
CONFIRMATION = "DELETE_UNSELECTED_WEIGHTS"


KEEP = (
    (
        "official_released_single_object_bc",
        "ActionDiffusion/bc/saved_models/"
        "1obj_seq2000_DexRep_pro100_start_uniform_vis_action_dsam_mod/"
        "last.ckpt",
    ),
    (
        "provided_architecture_same_data_warmstart_bc",
        "custom_tools/runs/bc/"
        "official_bc_scaled20_matched_seed2025_e100_v1/"
        "epoch=079-step=75440.ckpt",
    ),
    (
        "noise005_bc_seed2025_soup_source",
        "custom_tools/runs/bc/"
        "multicategory_bc_noise005_seed2025_e100/"
        "epoch=039-step=8640.ckpt",
    ),
    (
        "noise005_bc_seed2026_soup_source",
        "custom_tools/runs/bc/"
        "multicategory_bc_noise005_seed2026_e40/"
        "epoch=039-step=8640.ckpt",
    ),
    (
        "weighted_bc_soup",
        "custom_tools/runs/bc/model_soups/"
        "noise005_s2025_s2026_weighted2to1.ckpt",
    ),
    (
        "category_teacher_bottle",
        "custom_tools/runs/bc/"
        "category_expert_bottle_scaled20_noise005_soup_seed2025_e40_v1/"
        "epoch=029-step=8790.ckpt",
    ),
    (
        "category_teacher_mug",
        "custom_tools/runs/bc/"
        "category_expert_mug_scaled20_noise005_soup_seed2025_e40_v1/"
        "epoch=009-step=2250.ckpt",
    ),
    (
        "category_teacher_bowl",
        "custom_tools/runs/bc/"
        "category_expert_bowl_scaled20_noise005_soup_seed2025_e40_v1/"
        "epoch=039-step=7440.ckpt",
    ),
    (
        "category_teacher_camera",
        "custom_tools/runs/bc/"
        "category_expert_camera_scaled20_noise005_soup_seed2025_e40_v1/"
        "epoch=039-step=9520.ckpt",
    ),
    (
        "offline_taskid_student",
        "custom_tools/runs/bc/"
        "unified_student_taskid_scaled20_t100_seed2025_e20_v1/"
        "epoch=014-step=14145.ckpt",
    ),
    (
        "online_r1_taskid_student",
        "custom_tools/runs/bc/"
        "unified_student_taskid_online_r1_frac025_seed2025_e10_v1/"
        "epoch=001-step=2232.ckpt",
    ),
    (
        "temporal3_final",
        "custom_tools/runs/bc/"
        "unified_student_taskid_temporal3_seed2025_e4_v1/"
        "epoch=003-step=5152.ckpt",
    ),
    (
        "offline_student_without_taskid_ablation",
        "custom_tools/runs/bc/"
        "unified_student_notask_scaled20_t100_seed2025_e20_v1/"
        "epoch=004-step=4715.ckpt",
    ),
    (
        "online_r1_without_taskid_ablation",
        "custom_tools/runs/bc/"
        "unified_student_notask_online_r1_frac025_seed2025_e10_v1/"
        "epoch=009-step=11160.ckpt",
    ),
    (
        "temporal3_without_taskid_ablation",
        "custom_tools/runs/bc/"
        "unified_student_notask_temporal3_seed2025_e4_v1/"
        "epoch=001-step=2576.ckpt",
    ),
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gib(value):
    return value / (1024 ** 3)


def relative(path):
    return str(path.resolve().relative_to(ROOT.resolve()))


def collect_inventory():
    keep_rows = []
    keep_paths = set()
    missing = []
    for role, relative_path in KEEP:
        path = ROOT / relative_path
        if not path.is_file():
            missing.append(relative_path)
            continue
        resolved = path.resolve()
        keep_paths.add(resolved)
        keep_rows.append({
            "role": role,
            "path": relative(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    if missing:
        raise FileNotFoundError(
            "Required keep checkpoints are missing:\n{}".format(
                "\n".join(missing)))

    remove_paths = []
    if BC_ROOT.is_dir():
        for path in sorted(BC_ROOT.rglob("*.ckpt")):
            if path.resolve() not in keep_paths:
                remove_paths.append(path)
    if RESIDUAL_ROOT.is_dir():
        for path in sorted(RESIDUAL_ROOT.rglob("*")):
            if path.is_file() and path.suffix in {".pt", ".pth", ".ckpt"}:
                remove_paths.append(path)

    remove_rows = [{
        "path": relative(path),
        "size_bytes": path.stat().st_size,
    } for path in remove_paths]
    return keep_rows, remove_rows


def write_manifest(path, keep_rows, remove_rows, executed):
    kept_bytes = sum(row["size_bytes"] for row in keep_rows)
    removed_bytes = sum(row["size_bytes"] for row in remove_rows)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "executed" if executed else "preview",
        "policy": (
            "Keep final serial-mainline checkpoints and the three selected "
            "no-Task-ID controls. Remove only unselected model-weight files; "
            "preserve source, configs, logs, curves, and evaluation results."
        ),
        "kept_file_count": len(keep_rows),
        "kept_size_bytes": kept_bytes,
        "kept_size_gib": gib(kept_bytes),
        "removal_file_count": len(remove_rows),
        "removal_size_bytes": removed_bytes,
        "removal_size_gib": gib(removed_bytes),
        "kept_files": keep_rows,
        "removed_files" if executed else "planned_removals": remove_rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", default=str(DEFAULT_MANIFEST),
        help="Inventory output path.")
    parser.add_argument(
        "--execute", action="store_true",
        help="Actually unlink planned unselected weight files.")
    parser.add_argument(
        "--confirmation", default="",
        help="Required exact phrase for --execute.")
    args = parser.parse_args()

    if args.execute and args.confirmation != CONFIRMATION:
        raise ValueError(
            "--execute requires --confirmation {}".format(CONFIRMATION))

    manifest = Path(args.manifest).expanduser().resolve()
    keep_rows, remove_rows = collect_inventory()
    kept_bytes = sum(row["size_bytes"] for row in keep_rows)
    removal_bytes = sum(row["size_bytes"] for row in remove_rows)

    print("KEEP files={} size={:.2f} GiB".format(
        len(keep_rows), gib(kept_bytes)))
    print("REMOVE files={} size={:.2f} GiB".format(
        len(remove_rows), gib(removal_bytes)))

    if not args.execute:
        write_manifest(manifest, keep_rows, remove_rows, executed=False)
        print("PREVIEW_ONLY")
        print("manifest={}".format(manifest))
        print(
            "To execute: python custom_tools/storage_cleanup.py --execute "
            "--confirmation {}".format(CONFIRMATION))
        return

    for row in remove_rows:
        path = ROOT / row["path"]
        path.unlink()
    write_manifest(manifest, keep_rows, remove_rows, executed=True)
    print("CLEANUP_COMPLETE")
    print("manifest={}".format(manifest))


if __name__ == "__main__":
    main()
