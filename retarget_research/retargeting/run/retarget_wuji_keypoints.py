#!/usr/bin/env python3
"""用15对语义关键点把Shadow轨迹重定向到20自由度Wuji右手。

输入：包含`grasp_seqs`的Shadow `.npy`、轨迹索引和Wuji映射配置。
输出：`(N,70,26)`候选轨迹，其中每帧为手腕6维加20个手指关节。
内部逻辑：每帧用SLSQP联合优化20个独立关节和手腕位姿，并以上一帧热启动。
作用：补齐参考仓库缺少的Shadow→Wuji数学重定向入口，建立第三只手的基线。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import nlopt
import numpy as np
import torch
import transforms3d


RETARGET_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = RETARGET_ROOT.parent
REFERENCE_SCRIPTS = PROJECT_ROOT / "reference" / "HandRetargetTask2026" / "scripts"
THIRD_PARTY_PK = (
    PROJECT_ROOT
    / "reference"
    / "HandRetargetTask2026"
    / "third_party"
    / "pytorch_kinematics"
)
for path in (REFERENCE_SCRIPTS, THIRD_PARTY_PK):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from utils.HandModel_xhand import HandModel_xhand  # noqa: E402
from utils.hand_model import HandModel as ShadowHandModel  # noqa: E402
from utils.rot6d import robust_compute_orth6d_from_eulerXYZ  # noqa: E402


R_ALIGN = np.asarray(
    [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    dtype=np.float32,
)


def build_shadow_model():
    """创建只用于关键点目标的CPU Shadow运动学模型。

    输入：参考仓库中的Shadow MJCF、mesh和关键点文件。
    输出：启用21点定义的Shadow模型。
    内部逻辑：关闭不参与当前损失的表面采样，只保留正向运动学。
    作用：把28维源动作转换为每帧世界坐标关键点监督。
    """
    base = REFERENCE_SCRIPTS / "assets" / "mjcf_free"
    return ShadowHandModel(
        mjcf_path=str(base / "shadow_hand_vis_new.xml"),
        mesh_path=str(base / "meshes"),
        contact_points_path=str(base / "contact_points.json"),
        penetration_points_path=str(base / "penetration_points.json"),
        n_surface_points=0,
        device="cpu",
        use_joint21=True,
    )


def build_wuji_model():
    """创建20个关节全部独立可优化的Wuji右手模型。

    输入：参考仓库Wuji右手URDF、mesh和`penetration_wuji_right.json`。
    输出：CPU上的通用URDF手模型，关键点数26、关节参数数20。
    内部逻辑：复用参考通用手模型解析URDF，并允许缺少contact候选文件。
    作用：提供可求导的Wuji正向运动学、关节边界和世界关键点。
    """
    asset = REFERENCE_SCRIPTS / "assets" / "wujihand_urdf" / "urdf"
    return HandModel_xhand(
        robot_name="wuji_right",
        urdf_filename="right.urdf",
        mesh_path="../",
        batch_size=1,
        device=torch.device("cpu"),
        mesh_nsp=128,
        hand_scale=1.0,
        asset_dir=str(asset),
        allow_missing_contacts=True,
    )


def load_pairs(mapping_config):
    """读取并返回Shadow↔Wuji的15对语义关键点。

    输入：Wuji映射JSON路径。
    输出：保持配置顺序的pair字典列表。
    内部逻辑：读取JSON并拒绝不是15对的未完成配置。
    作用：让优化器索引来自可审查配置，而非散落在代码中的数字数组。
    """
    config = json.loads(
        Path(mapping_config).read_text()
    )
    pairs = config["pairs"]
    if len(pairs) != 15:
        raise ValueError("Wuji当前基线必须使用15对语义点")
    return pairs


def shadow_keypoints(frames, model):
    """批量恢复一条Shadow轨迹的21个世界关键点。

    输入：`(T,28)`源帧和Shadow模型。
    输出：`(T,21,3)`NumPy数组。
    内部逻辑：把欧拉角改写成旋转6D，拼为模型需要的31维参数后正向运动学。
    作用：为Wuji每一帧提供目标，而不要求数据集含有Wuji关节标签。
    """
    source = torch.as_tensor(frames, dtype=torch.float32)
    q = torch.zeros((len(source), 31), dtype=torch.float32)
    q[:, :3] = source[:, :3]
    q[:, 3:9] = robust_compute_orth6d_from_eulerXYZ(source[:, 3:6])
    q[:, 9:] = source[:, 6:]
    model.set_parameters(q)
    return model.get_penetraion_keypoints().detach().cpu().numpy()


def clip_start_to_bounds(values, lower, upper, epsilon=1e-6):
    """把SLSQP初值夹到关节和位姿边界内侧。

    输入：同形状的初值、下界、上界和安全间隔。
    输出：严格位于边界内侧的64位数组。
    内部逻辑：逐维执行`np.clip(lower+epsilon, upper-epsilon)`。
    作用：避免欧拉角或关节在浮点舍入后被NLopt判定为非法初值。
    """
    return np.clip(
        np.asarray(values, dtype=np.float64),
        np.asarray(lower, dtype=np.float64) + epsilon,
        np.asarray(upper, dtype=np.float64) - epsilon,
    )


def initial_values(shadow_frame, joint_count):
    """用Shadow手腕构造Wuji首帧优化初值。

    输入：一帧28维Shadow动作和Wuji关节数。
    输出：`[零关节20, Shadow平移3, 对齐欧拉角3]`。
    内部逻辑：用固定坐标旋转左乘Shadow手腕旋转，手指从零姿态开始。
    作用：让非线性优化从正确手掌方向附近开始，后续帧则使用上一帧结果。
    """
    shadow_rotation = transforms3d.euler.euler2mat(
        *[float(value) for value in shadow_frame[3:6]], axes="sxyz"
    )
    aligned_rotation = R_ALIGN @ shadow_rotation
    euler = np.asarray(
        transforms3d.euler.mat2euler(aligned_rotation, axes="sxyz"),
        dtype=np.float32,
    )
    return np.concatenate(
        [
            np.zeros(joint_count, dtype=np.float32),
            np.asarray(shadow_frame[:3], dtype=np.float32),
            euler,
        ]
    )


class WujiFrameObjective:
    """计算单帧Wuji语义点误差和可选时间连续性损失。"""

    def __init__(
        self,
        model,
        target_world,
        target_indices,
        previous_values=None,
        joint_temporal_weight=0.0,
        translation_temporal_weight=0.0,
        rotation_temporal_weight=0.0,
        contact_indices=None,
        contact_targets=None,
        contact_weight=0.0,
        reference_values=None,
        reference_joint_weight=0.0,
        reference_translation_weight=0.0,
        reference_rotation_weight=0.0,
    ):
        """保存单帧目标、目标索引和上一帧姿态。

        输入：模型、15点目标、时序项，以及可选接触锚点和参考姿态项。
        输出：无返回；初始化可被NLopt重复调用的目标对象。
        内部逻辑：把固定数组转为CPU张量，并从模型读取20维关节数。
        作用：隔离每次损失求值不变的数据，保持优化循环清楚。
        """
        self.model = model
        self.target = torch.as_tensor(target_world, dtype=torch.float32)
        self.target_indices = np.asarray(target_indices, dtype=np.int64)
        self.joint_count = len(model.robot.get_joint_parameter_names())
        self.previous = (
            None
            if previous_values is None
            else torch.as_tensor(previous_values, dtype=torch.float32)
        )
        self.joint_temporal_weight = float(joint_temporal_weight)
        self.translation_temporal_weight = float(translation_temporal_weight)
        self.rotation_temporal_weight = float(rotation_temporal_weight)
        self.contact_indices = (
            None
            if contact_indices is None
            else np.asarray(contact_indices, dtype=np.int64)
        )
        self.contact_targets = (
            None
            if contact_targets is None
            else torch.as_tensor(contact_targets, dtype=torch.float32)
        )
        self.contact_weight = float(contact_weight)
        self.reference = (
            None
            if reference_values is None
            else torch.as_tensor(reference_values, dtype=torch.float32)
        )
        self.reference_joint_weight = float(reference_joint_weight)
        self.reference_translation_weight = float(reference_translation_weight)
        self.reference_rotation_weight = float(reference_rotation_weight)

    def __call__(self, values, gradient=None):
        """返回一组26维候选姿态的可求导标量损失。

        输入：`[关节20,平移3,欧拉角3]`和可选NLopt梯度缓冲区。
        输出：关键点均方平方距离乘1000，加三类相邻帧正则。
        内部逻辑：欧拉角转旋转6D，计算15点、可选接触锚点和连续/参考损失。
        作用：告诉SLSQP候选姿态的好坏及下降方向。
        """
        values_tensor = torch.tensor(
            np.asarray(values, dtype=np.float32),
            dtype=torch.float32,
            requires_grad=True,
        )
        joint_count = self.joint_count
        joints = values_tensor[:joint_count].view(1, joint_count)
        translation = values_tensor[joint_count : joint_count + 3].view(1, 3)
        euler = values_tensor[joint_count + 3 : joint_count + 6].view(1, 3)
        rotation = robust_compute_orth6d_from_eulerXYZ(euler)
        q = torch.cat([translation, rotation, joints], dim=1)
        all_points = self.model.get_penetraion_keypoints(q=q)[0]
        points = all_points[self.target_indices]
        difference = points - self.target
        loss = torch.mean(torch.sum(difference * difference, dim=1)) * 1000.0
        if self.contact_indices is not None and self.contact_weight > 0:
            contact_difference = (
                all_points[self.contact_indices] - self.contact_targets
            )
            loss = loss + self.contact_weight * torch.mean(
                torch.sum(contact_difference * contact_difference, dim=1)
            ) * 1000.0
        if self.previous is not None:
            previous_rotation = robust_compute_orth6d_from_eulerXYZ(
                self.previous[joint_count + 3 : joint_count + 6].view(1, 3)
            )
            loss = loss + self.joint_temporal_weight * torch.mean(
                (values_tensor[:joint_count] - self.previous[:joint_count]) ** 2
            )
            loss = loss + self.translation_temporal_weight * torch.mean(
                (
                    values_tensor[joint_count : joint_count + 3]
                    - self.previous[joint_count : joint_count + 3]
                )
                ** 2
            )
            loss = loss + self.rotation_temporal_weight * torch.mean(
                (rotation - previous_rotation) ** 2
            )
        if self.reference is not None:
            reference_rotation = robust_compute_orth6d_from_eulerXYZ(
                self.reference[joint_count + 3 : joint_count + 6].view(1, 3)
            )
            loss = loss + self.reference_joint_weight * torch.mean(
                (values_tensor[:joint_count] - self.reference[:joint_count]) ** 2
            )
            loss = loss + self.reference_translation_weight * torch.mean(
                (
                    values_tensor[joint_count : joint_count + 3]
                    - self.reference[joint_count : joint_count + 3]
                )
                ** 2
            )
            loss = loss + self.reference_rotation_weight * torch.mean(
                (rotation - reference_rotation) ** 2
            )
        if gradient is not None and len(gradient) > 0:
            loss.backward()
            gradient[:] = values_tensor.grad.detach().numpy().astype(np.float64)
        return float(loss.detach().item())


def retarget_trajectory(
    source_frames,
    model,
    target_points,
    target_indices,
    maxeval,
    translation_bound,
    joint_temporal_weight,
    translation_temporal_weight,
    rotation_temporal_weight,
    progress_prefix=None,
):
    """逐帧优化一条Shadow轨迹对应的Wuji姿态。

    输入：源帧、Wuji模型、15点目标/索引、优化边界、时序权重和进度前缀。
    输出：内部顺序`(T,26)`轨迹和`(T,)`最终损失。
    内部逻辑：每帧新建有界SLSQP；首帧对齐初始化，之后以上一帧热启动。
    作用：把单帧目标组合成连续可保存的完整候选轨迹。
    """
    joint_lower = model.revolute_joints_q_lower[0].detach().numpy()
    joint_upper = model.revolute_joints_q_upper[0].detach().numpy()
    joint_count = len(joint_lower)
    lower = np.concatenate(
        [joint_lower, np.full(3, -translation_bound), np.full(3, -np.pi)]
    )
    upper = np.concatenate(
        [joint_upper, np.full(3, translation_bound), np.full(3, np.pi)]
    )
    results, losses = [], []
    previous = None
    for frame_index, source_frame in enumerate(source_frames):
        start = (
            initial_values(source_frame, joint_count)
            if previous is None
            else previous.copy()
        )
        start = clip_start_to_bounds(start, lower, upper)
        objective = WujiFrameObjective(
            model,
            target_points[frame_index],
            target_indices,
            previous_values=previous,
            joint_temporal_weight=joint_temporal_weight,
            translation_temporal_weight=translation_temporal_weight,
            rotation_temporal_weight=rotation_temporal_weight,
        )
        optimizer = nlopt.opt(nlopt.LD_SLSQP, joint_count + 6)
        optimizer.set_min_objective(objective)
        optimizer.set_lower_bounds(lower.tolist())
        optimizer.set_upper_bounds(upper.tolist())
        optimizer.set_maxeval(maxeval)
        optimizer.set_xtol_rel(1e-6)
        optimizer.set_ftol_rel(1e-8)
        try:
            result = optimizer.optimize(start)
        except (nlopt.RoundoffLimited, RuntimeError):
            result = start
        previous = np.asarray(result, dtype=np.float32)
        results.append(previous)
        losses.append(objective(previous))
        completed = frame_index + 1
        if progress_prefix and (completed % 10 == 0 or completed == len(source_frames)):
            print(
                f"{progress_prefix}: frames={completed}/{len(source_frames)} "
                f"loss={losses[-1]:.6f}",
                flush=True,
            )
    return np.stack(results), np.asarray(losses, dtype=np.float32)


def retarget_file(args):
    """读取源数据、运行Wuji优化并保存标准候选文件。

    输入：命令行source/output/索引、优化次数、边界和时序权重。
    输出：包含候选、源索引、物体姿态、语义和方法元数据的`.npy`。
    内部逻辑：为源Z加参考0.4 m偏移，恢复Shadow点，逐轨迹优化并重排维度。
    作用：提供可复现、可供独立几何/物理评估读取的文件级入口。
    """
    source_data = np.load(args.source, allow_pickle=True).item()
    indices = args.trajectory_indices or [0]
    source_frames = np.asarray(
        source_data["grasp_seqs"][indices], dtype=np.float32
    ).copy()
    source_frames[:, :, 2] += args.source_z_offset
    shadow_model = build_shadow_model()
    wuji_model = build_wuji_model()
    pairs = load_pairs(args.mapping_config)
    shadow_indices = [pair["shadow_index"] for pair in pairs]
    wuji_indices = [pair["wuji_index"] for pair in pairs]
    outputs, all_losses = [], []
    for trajectory_index, trajectory in enumerate(source_frames):
        target_points = shadow_keypoints(trajectory, shadow_model)[:, shadow_indices]
        internal, losses = retarget_trajectory(
            trajectory,
            wuji_model,
            target_points,
            wuji_indices,
            args.maxeval,
            args.translation_bound,
            args.joint_temporal_weight,
            args.translation_temporal_weight,
            args.rotation_temporal_weight,
            progress_prefix=(
                f"trajectory={trajectory_index + 1}/{len(source_frames)} "
                f"source_index={indices[trajectory_index]}"
            ),
        )
        # 内部为关节20+手腕6；保存统一为手腕6+关节20。
        outputs.append(np.concatenate([internal[:, 20:26], internal[:, :20]], axis=1))
        all_losses.append(losses)
    output_frames = np.stack(outputs).astype(np.float32)
    loss_frames = np.stack(all_losses).astype(np.float32)
    output = {
        "grasp_seqs": output_frames,
        "optimization_loss_per_frame": loss_frames,
        "source_trajectory_indices": np.asarray(indices, dtype=np.int64),
        "obj_rotmat": np.asarray(source_data["obj_rotmat"])[indices],
        "obj_scale": np.asarray(source_data["obj_scale"])[indices],
        "mapping_semantics": [pair["semantic"] for pair in pairs],
        "mapping_config": str(args.mapping_config.resolve()),
        "wuji_joint_names": wuji_model.robot.get_joint_parameter_names(),
        "source_z_offset": float(args.source_z_offset),
        "maxeval": int(args.maxeval),
        "joint_temporal_weight": float(args.joint_temporal_weight),
        "translation_temporal_weight": float(args.translation_temporal_weight),
        "rotation_temporal_weight": float(args.rotation_temporal_weight),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, output, allow_pickle=True)
    print(f"trajectories={len(output_frames)}")
    print(f"frames={output_frames.shape[1]}")
    print(f"output_shape={output_frames.shape}")
    print(f"mean_final_loss={loss_frames.mean():.6f}")
    print(f"max_final_loss={loss_frames.max():.6f}")
    print(f"output={args.output}")


def main():
    """解析参数并运行Wuji单文件关键点基线。

    输入：源/输出路径、轨迹索引、SLSQP次数、位姿边界和时序权重。
    输出：目标npy及终端形状/损失摘要。
    内部逻辑：构造参数后交给`retarget_file`，不在入口隐藏实验默认值。
    作用：作为run分区中Wuji重定向的标准命令。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trajectory-indices", type=int, nargs="*")
    parser.add_argument("--maxeval", type=int, default=100)
    parser.add_argument("--translation-bound", type=float, default=2.0)
    parser.add_argument("--source-z-offset", type=float, default=0.4)
    parser.add_argument(
        "--mapping-config",
        type=Path,
        default=RETARGET_ROOT / "configs" / "wuji_keypoint_map.json",
    )
    parser.add_argument("--joint-temporal-weight", type=float, default=0.0)
    parser.add_argument("--translation-temporal-weight", type=float, default=0.0)
    parser.add_argument("--rotation-temporal-weight", type=float, default=0.0)
    retarget_file(parser.parse_args())


if __name__ == "__main__":
    main()
