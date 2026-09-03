#!/usr/bin/env python3
"""GeoRT式学习重定向：用精确可微FK学习五指尖到目标手关节角的映射。"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation
import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retarget_research.minimal_impl.config import LINKER_POINTS
from retarget_research.minimal_impl.data import load_npy
from retarget_research.minimal_impl.kinematics import (
    build_shadow_model,
    build_target_model,
    shadow_keypoints,
)


TIP_INDICES = {
    "linker": (14, 3, 6, 9, 12),
    "xhand": (8, 14, 20, 25, 29),
    "wuji": (5, 10, 15, 20, 25),
}
FINGER_JOINTS = {
    "linker": ((0, 1), (2,), (3,), (4,), (5,)),
    "xhand": ((0, 1, 2), (3, 4, 5), (6, 7), (8, 9), (10, 11)),
    "wuji": ((0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11),
             (12, 13, 14, 15), (16, 17, 18, 19)),
}
SOURCE_TIPS = (20, 4, 8, 12, 16)
TARGET_PALM_FRAME = {
    "linker": (0, 1, 4, 7),
    "xhand": (0, 10, 15, 22),
    "wuji": (0, 6, 11, 16),
}


class FingerIK(nn.Module):
    """输入五个源指尖的掌坐标，输出归一化后的目标手主动关节角。"""

    def __init__(self, hand, hidden=128):
        super().__init__()
        self.groups = FINGER_JOINTS[hand]
        self.nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(3, hidden), nn.LeakyReLU(),
                nn.Linear(hidden, hidden), nn.LeakyReLU(),
                nn.Linear(hidden, len(group)), nn.Tanh(),
            )
            for group in self.groups
        ])
        self.dof = sum(map(len, self.groups))

    def forward(self, tips):
        output = tips.new_zeros((len(tips), self.dof))
        for finger, (network, group) in enumerate(zip(self.nets, self.groups)):
            output[:, list(group)] = network(tips[:, finger])
        return output


def target_points_fk(model, hand, joints):
    """输入实际主动关节角，输出目标手全部预定义点在URDF基座下的位置。"""
    batch = len(joints)
    rotation6d = joints.new_tensor((1.0, 0.0, 0.0, 0.0, 1.0, 0.0)).expand(batch, -1)
    q = torch.cat((joints.new_zeros((batch, 3)), rotation6d, joints), dim=1)
    if hand != "linker":
        return model.get_penetraion_keypoints(q=q)
    model.update_kinematics(q)
    points = []
    for link_name, xyz in LINKER_POINTS:
        local = joints.new_tensor(xyz).reshape(1, 1, 3).expand(batch, -1, -1)
        points.append(model.current_status[link_name].transform_points(local)[:, 0])
    return torch.stack(points, dim=1)


def palm_frame(points, indices):
    """按GeoRT约定建立掌坐标：+Y朝拇指侧，+Z朝中指根部，+X为掌面法向。"""
    palm, index_base, middle_base, ring_base = (points[:, i] for i in indices)
    z_axis = F.normalize(middle_base - palm, dim=-1)
    y_aux = F.normalize(index_base - ring_base, dim=-1)
    x_axis = F.normalize(torch.cross(y_aux, z_axis, dim=-1), dim=-1)
    y_axis = F.normalize(torch.cross(z_axis, x_axis, dim=-1), dim=-1)
    return palm, torch.stack((x_axis, y_axis, z_axis), dim=-1)


def target_canonical_frame(model, hand, lower, upper):
    """从目标手零位结构求一次固定掌坐标，供所有目标工作空间样本共用。"""
    zero = torch.minimum(torch.maximum(torch.zeros_like(lower), lower), upper).unsqueeze(0)
    points = target_points_fk(model, hand, zero)
    origin, frame = palm_frame(points, TARGET_PALM_FRAME[hand])
    return origin[0].detach(), frame[0].detach()


def anatomical_limits(model, hand):
    """读取URDF边界，并对Wuji普通指PIP/DIP应用已验证的防反弯下界。"""
    lower = model.revolute_joints_q_lower[0].detach().clone()
    upper = model.revolute_joints_q_upper[0].detach().clone()
    if hand == "wuji":
        names = list(model.robot.get_joint_parameter_names())
        for finger in range(2, 6):
            for joint in (3, 4):
                index = names.index(f"finger{finger}_joint{joint}")
                lower[index] = max(float(lower[index]), -0.08726646)
    return lower, upper


def target_tip_fk(model, hand, normalized_joints, lower, upper, origin=None, frame=None):
    """输入归一化主动关节角，输出目标手五指尖在基座坐标系中的精确FK位置。"""
    joints = lower + 0.5 * (normalized_joints + 1.0) * (upper - lower)
    tips = target_points_fk(model, hand, joints)[:, list(TIP_INDICES[hand])]
    if origin is not None:
        tips = torch.matmul(tips - origin, frame)
    return tips


def audit_records(audit_path, split=None, one_per_category=False):
    """读取固定审计记录；可选择split，并按类别只保留第一条。"""
    rows = json.loads(Path(audit_path).read_text(encoding="utf-8"))["results"]
    rows = sorted(rows, key=lambda x: (x["category"], x["object_name"], x["source_trajectory_index"]))
    if split:
        rows = [row for row in rows if row["evaluation_split"] == split]
    if one_per_category:
        selected = {}
        for row in rows:
            selected.setdefault(row["category"], row)
        rows = list(selected.values())
    return rows


def source_tip_dataset(audit_path, cache_path):
    """按GeoRT掌坐标规范化Shadow关键点，输出(N,5,3)五指尖训练点。"""
    cache_path = Path(cache_path)
    if cache_path.exists():
        return np.load(cache_path)
    shadow = build_shadow_model()
    groups = {}
    for row in audit_records(audit_path):
        report = json.loads(Path(row["geometry_report"]).read_text(encoding="utf-8"))
        groups.setdefault(report["source"], []).append(
            int(row["source_trajectory_index"])
        )
    result = []
    for source_path, indices in groups.items():
        source = load_npy(source_path)
        source_frames = np.asarray(source["grasp_seqs"])[indices].reshape(-1, 28).copy()
        points = torch.from_numpy(shadow_keypoints(source_frames, shadow))
        origin, frame = palm_frame(points, (0, 1, 5, 9))
        canonical = torch.matmul(points - origin[:, None], frame)
        result.append(canonical[:, SOURCE_TIPS].numpy().astype(np.float32))
    points = np.concatenate(result)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, points)
    return points


def workspace(model, hand, lower, upper, count, batch_size, device, origin, frame):
    """均匀采样目标手关节范围并用精确FK生成五指工作空间。"""
    generator = torch.Generator(device=device).manual_seed(20260827)
    result = []
    for start in range(0, count, batch_size):
        size = min(batch_size, count - start)
        q = torch.rand((size, len(lower)), generator=generator, device=device) * 2.0 - 1.0
        with torch.no_grad():
            result.append(target_tip_fk(model, hand, q, lower, upper, origin, frame).cpu())
    return torch.cat(result)


def chamfer_by_finger(predicted, target):
    """逐指比较两批工作空间点集，避免不同手指之间错误匹配。"""
    total = predicted.new_tensor(0.0)
    for finger in range(5):
        distance = torch.cdist(predicted[:, finger], target[:, finger]).square()
        total = total + distance.min(1).values.mean() + distance.min(0).values.mean()
    return total


def train(args):
    """按GeoRT的工作空间、局部方向、曲率和捏合约束训练IK网络。"""
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    source = torch.from_numpy(source_tip_dataset(args.audit, args.source_cache))
    model = build_target_model(args.hand, device=device)
    lower, upper = anatomical_limits(model, args.hand)
    origin, frame = target_canonical_frame(model, args.hand, lower, upper)
    target_space = workspace(model, args.hand, lower, upper, args.workspace_samples,
                             args.batch_size, device, origin, frame)
    network = FingerIK(args.hand, args.hidden).to(device)
    if args.init_checkpoint is not None:
        initial = torch.load(args.init_checkpoint, map_location=device)
        network.load_state_dict(initial["model"])
    optimizer = torch.optim.AdamW(network.parameters(), lr=args.learning_rate)
    source_reach = torch.quantile(torch.linalg.vector_norm(source, dim=-1), 0.95, dim=0)
    target_reach = torch.quantile(torch.linalg.vector_norm(target_space, dim=-1), 0.95, dim=0)
    semantic_scale = torch.median(target_reach / source_reach).to(device)

    history = []
    for epoch in range(1, args.epochs + 1):
        order = torch.randperm(len(source))[:args.epoch_samples]
        totals = []
        for start in range(0, len(order), args.batch_size):
            points = source[order[start:start + args.batch_size]].to(device)
            if len(points) < 2:
                continue
            q = network(points)
            embedded = target_tip_fk(model, args.hand, q, lower, upper, origin, frame)
            semantic = (embedded - points * semantic_scale).square().mean()
            target = target_space[torch.randint(len(target_space), (len(points),))].to(device)
            chamfer = chamfer_by_finger(embedded, target)

            direction = F.normalize(torch.randn_like(points), dim=-1)
            delta = direction * (0.001 + 0.009 * torch.rand((len(points), 1, 1), device=device))
            embedded_delta = target_tip_fk(model, args.hand, network(points + delta), lower, upper,
                                           origin, frame)
            direction_loss = -(F.normalize(delta.reshape(-1, 3), dim=-1) *
                               F.normalize((embedded_delta - embedded).reshape(-1, 3), dim=-1)).sum(-1).mean()

            small = F.normalize(torch.randn_like(points), dim=-1) * 0.002
            positive = target_tip_fk(model, args.hand, network(points + small), lower, upper,
                                     origin, frame)
            negative = target_tip_fk(model, args.hand, network(points - small), lower, upper,
                                     origin, frame)
            curvature = (positive + negative - 2.0 * embedded).square().mean()

            pinch = embedded.new_tensor(0.0)
            for i in range(5):
                for j in range(i + 1, 5):
                    mask = torch.linalg.vector_norm(points[:, i] - points[:, j], dim=-1) < 0.015
                    if mask.any():
                        pinch = pinch + (embedded[mask, i] - embedded[mask, j]).square().sum(-1).mean()
            loss = direction_loss + args.chamfer_weight * chamfer + \
                   args.curvature_weight * curvature + args.pinch_weight * pinch + \
                   args.semantic_weight * semantic
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            totals.append((loss.item(), chamfer.item(), direction_loss.item(),
                           pinch.item(), semantic.item()))
        mean = np.mean(totals, axis=0)
        history.append({"epoch": epoch, "loss": float(mean[0]), "chamfer": float(mean[1]),
                        "direction": float(mean[2]), "pinch": float(mean[3]),
                        "semantic": float(mean[4])})
        print(f"epoch={epoch}/{args.epochs} loss={mean[0]:.6f} chamfer={mean[1]:.6f} "
              f"direction={mean[2]:.6f} pinch={mean[3]:.6f} semantic={mean[4]:.6f}",
              flush=True)

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "method": ("geort_semantic_anchor_v3" if args.semantic_weight > 0
                   else "geort_canonical_exact_fk_v2"),
        "hand": args.hand,
        "model": network.state_dict(), "hidden": args.hidden,
        "lower": lower.cpu(), "upper": upper.cpu(),
        "source_cache": str(Path(args.source_cache).resolve()),
        "history": history, "target_origin": origin.cpu(), "target_frame": frame.cpu(),
        "semantic_scale": float(semantic_scale), "semantic_weight": args.semantic_weight,
    }, args.checkpoint)


def apply(args):
    """用训练好的IK替换基线手指角，保留其手腕轨迹并生成固定50条候选。"""
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    network = FingerIK(args.hand, checkpoint["hidden"]).to(device)
    network.load_state_dict(checkpoint["model"])
    network.eval()
    shadow = build_shadow_model()
    rows = audit_records(args.audit, args.split, args.one_per_category)
    formal = json.loads(args.formal_manifest.read_text(encoding="utf-8"))
    assets = {entry["object_name"]: entry["object_asset_path"] for entry in formal["entries"]}
    entries = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for number, row in enumerate(rows, 1):
        report = json.loads(Path(row["geometry_report"]).read_text(encoding="utf-8"))
        source = load_npy(report["source"])
        baseline = load_npy(report["target"])
        source_index = int(row["source_trajectory_index"])
        target_rows = {int(v): i for i, v in enumerate(baseline["source_trajectory_indices"])}
        baseline_row = target_rows[source_index]
        source_frames = np.asarray(source["grasp_seqs"])[source_index].copy()
        points = torch.from_numpy(shadow_keypoints(source_frames, shadow))
        origin, frame = palm_frame(points, (0, 1, 5, 9))
        canonical = torch.matmul(points - origin[:, None], frame)
        wrist = np.asarray(baseline["grasp_seqs"])[baseline_row, :, :6]
        with torch.no_grad():
            normalized = network(canonical[:, SOURCE_TIPS].float().to(device))
            lower = checkpoint["lower"].to(device)
            upper = checkpoint["upper"].to(device)
            joints = lower + 0.5 * (normalized + 1.0) * (upper - lower)
        frames = np.concatenate((wrist, joints.cpu().numpy()), axis=1).astype(np.float32)
        baseline_count = len(baseline["source_trajectory_indices"])
        output = {
            key: (value[[baseline_row]].copy()
                  if isinstance(value, np.ndarray) and value.ndim and len(value) == baseline_count
                  else value)
            for key, value in baseline.items()
        }
        output.update({
            "grasp_seqs": frames[None],
            "source_trajectory_indices": np.asarray([source_index]),
            "obj_rotmat": np.asarray(source["obj_rotmat"])[[source_index]],
            "obj_scale": np.asarray(source["obj_scale"])[[source_index]],
            "method": checkpoint["method"],
        })
        np.save(args.output_dir / f"{row['object_name']}.npy", output, allow_pickle=True)
        entries.append({
            "object_name": row["object_name"], "category": row["category"],
            "source_path": report["source"],
            "object_asset_path": assets[row["object_name"]],
            "trajectory_indices": [source_index], "calibration_indices": [source_index],
            "heldout_indices": [],
        })
        print(f"[{args.hand}] {number}/{len(rows)} {row['object_name']}", flush=True)
    manifest = {"schema_version": 1, "purpose": "GeoRT exact-FK calibration screen",
                "hand": args.hand, "trajectory_count": len(entries), "entries": entries}
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("train", "apply"))
    parser.add_argument("--hand", choices=("linker", "xhand", "wuji"), required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-cache", type=Path)
    parser.add_argument("--workspace-samples", type=int, default=20000)
    parser.add_argument("--epoch-samples", type=int, default=20000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--chamfer-weight", type=float, default=80.0)
    parser.add_argument("--curvature-weight", type=float, default=0.1)
    parser.add_argument("--pinch-weight", type=float, default=1.0)
    parser.add_argument("--semantic-weight", type=float, default=0.0)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--split", default="calibration")
    parser.add_argument("--one-per-category", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--manifest-output", type=Path)
    parser.add_argument("--formal-manifest", type=Path)
    args = parser.parse_args()
    if args.mode == "train":
        train(args)
    else:
        apply(args)


if __name__ == "__main__":
    main()
