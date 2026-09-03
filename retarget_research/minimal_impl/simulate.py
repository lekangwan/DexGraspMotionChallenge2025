"""Isaac Gym轨迹重放、稳定运输判定和批量候选评估。

输入：一组已经重定向好的目标手Case。
输出：物体位置/接触曲线、参考成功和稳定运输指标。
内部逻辑：70帧各插值3个60 Hz物理步，末帧保持30步。
发挥作用：给CEM返回真实物理分数，并独立验证最终轨迹。
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

from .config import REFERENCE_SCRIPTS, WRIST_LINKER_WUJI, WRIST_XHAND, XHAND_FINGERS
from .data import cases_from_files


def _isaac():
    """延迟导入Isaac Gym；输入无，输出gymapi；使纯数学测试不需启动仿真。"""
    from isaacgym import gymapi
    return gymapi


def create_sim(dt=1.0 / 60.0, substeps=2):
    """输入物理步长/子步数，输出Gym和sim；统一重力、PhysX求解器和地面参数。"""
    gymapi = _isaac()
    gym = gymapi.acquire_gym()
    params = gymapi.SimParams()
    params.dt, params.substeps = float(dt), int(substeps)
    params.up_axis = gymapi.UP_AXIS_Z
    params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    params.use_gpu_pipeline = False
    params.physx.use_gpu = False
    params.physx.solver_type = 1
    params.physx.num_position_iterations = 8
    params.physx.num_velocity_iterations = 1
    params.physx.contact_offset = 0.002
    sim = gym.create_sim(0, -1, gymapi.SIM_PHYSX, params)
    plane = gymapi.PlaneParams()
    plane.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    gym.add_ground(sim, plane)
    return gym, sim


def _drive_gains(names, finger_stiffness=120.0, finger_damping=5.0):
    """输入DOF名称，输出刚度/阻尼；手腕硬以跟踪位姿，手指柔以允许接触。"""
    translation = set(WRIST_LINKER_WUJI[:3] + WRIST_XHAND[:3])
    rotation = set(WRIST_LINKER_WUJI[3:] + WRIST_XHAND[3:])
    stiffness, damping = [], []
    for name in names:
        stiffness.append(20000.0 if name in translation else 2000.0 if name in rotation else finger_stiffness)
        damping.append(500.0 if name in translation else 80.0 if name in rotation else finger_damping)
    return np.asarray(stiffness), np.asarray(damping)


def load_hand_asset(gym, sim, hand):
    """输入Gym/sim/手名，输出手asset、DOF属性/名称；加载带虚拟6D手腕的URDF。"""
    gymapi = _isaac()
    options = gymapi.AssetOptions()
    options.fix_base_link = True
    options.disable_gravity = True
    options.collapse_fixed_joints = False
    options.default_dof_drive_mode = int(gymapi.DOF_MODE_POS)
    options.use_mesh_materials = True
    if hand == "linker":
        root, filename = REFERENCE_SCRIPTS / "assets/linkerhand/o6/right", "linkerhand_o6_right6d.urdf"
    elif hand == "xhand":
        root, filename = REFERENCE_SCRIPTS / "assets/xhand", "xhand_euler_control.urdf"
    else:
        root, filename = REFERENCE_SCRIPTS / "assets/wujihand_urdf/urdf", "right6d.urdf"
    asset = gym.load_asset(sim, str(root), filename, options)
    names = list(gym.get_asset_dof_names(asset))
    properties = gym.get_asset_dof_properties(asset)
    properties["driveMode"].fill(int(gymapi.DOF_MODE_POS))
    properties["stiffness"][:], properties["damping"][:] = _drive_gains(names)
    return asset, properties, names


def map_command(hand, command, dof_names, metadata):
    """输入手名、12/18/26维命令和DOF名称，输出URDF顺序目标；Linker在此展开mimic。"""
    command = np.asarray(command, dtype=np.float32)
    if hand == "linker":
        active = command[6:]
        values = dict(zip(WRIST_LINKER_WUJI, command[:6]))
        values.update({
            "rh_thumb_cmc_yaw": active[0], "rh_thumb_cmc_pitch": active[1],
            "rh_thumb_ip": active[1] * 1.86,
            "rh_index_mcp_pitch": active[2], "rh_index_dip": active[2] * 0.89,
            "rh_middle_mcp_pitch": active[3], "rh_middle_dip": active[3] * 0.89,
            "rh_ring_mcp_pitch": active[4], "rh_ring_dip": active[4] * 0.89,
            "rh_pinky_mcp_pitch": active[5], "rh_pinky_dip": active[5] * 0.89,
        })
    elif hand == "xhand":
        values = dict(zip(WRIST_XHAND + XHAND_FINGERS, command))
    else:
        values = dict(zip(WRIST_LINKER_WUJI + tuple(metadata["wuji_joint_names"]), command))
    return np.asarray([values[name] for name in dof_names], dtype=np.float32)


def load_object_asset(gym, sim, object_dir):
    """输入COACD物体目录，输出动态asset；在PhysX中使用真实碰撞网格。"""
    gymapi = _isaac()
    options = gymapi.AssetOptions()
    options.density = 100.0
    options.override_com = True
    options.override_inertia = True
    options.use_mesh_materials = True
    return gym.load_asset(sim, str(Path(object_dir) / "coacd"), "coacd_1.urdf", options)


def object_pose(object_dir, scale, rotation):
    """输入网格/缩放/旋转，输出初始Transform；旋转缩放后将最低点放在地面上方5 mm。"""
    gymapi = _isaac()
    mesh = trimesh.load_mesh(Path(object_dir) / "coacd/decomposed.obj", process=False)
    vertices = np.asarray(mesh.vertices) @ np.asarray(rotation).T * float(scale)
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0.0, -float(vertices[:, 2].min()) + 0.005)
    pose.r = gymapi.Quat(*map(float, Rotation.from_matrix(rotation).as_quat()))
    return pose


def actor_indices(gym, env, actor):
    """输入actor，输出环境域刚体索引集合；用于只统计手物接触。"""
    gymapi = _isaac()
    return {gym.get_actor_rigid_body_index(env, actor, i, gymapi.DOMAIN_ENV)
            for i in range(gym.get_actor_rigid_body_count(env, actor))}


def pair_contact_count(contacts, hand_indices, object_indices):
    """输入接触记录和两组刚体索引，输出手物接触数；兼容body0/body1交换。"""
    return sum((int(c["body0"]) in hand_indices and int(c["body1"]) in object_indices)
               or (int(c["body1"]) in hand_indices and int(c["body0"]) in object_indices)
               for c in contacts)


def longest_true_run(mask):
    """输入布尔序列，输出最长连续True步数；排除瞬间越过高度阈值。"""
    best = current = 0
    for value in np.asarray(mask, dtype=bool):
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def success_metrics(positions, initial, contacts, hand_poses=None, object_quaternions=None):
    """输入物体轨迹/接触/手腕位姿，输出参考成功、15 cm稳定抬升和掌物不滑移指标。"""
    positions, contacts = np.asarray(positions), np.asarray(contacts)
    lift = positions[:, 2] - initial[2]
    drift = np.linalg.norm(positions[:, :2] - initial[None, :2], axis=1)
    reference = ((np.linalg.norm(positions - np.array([0.0, 0.0, 0.30]), axis=1) <= 0.15)
                 | (positions[:, 2] >= 0.30))
    valid = (lift >= 0.15) & (drift <= 0.25)
    stable = bool(longest_true_run(valid) >= 30 and valid[-30:].all()
                  and lift.max() - lift[-1] <= 0.03
                  and np.ptp(lift[-30:]) <= 0.01 and (contacts[-30:] > 0).all())
    transport = stable
    translation_change = rotation_change = None
    if hand_poses is not None and object_quaternions is not None:
        hand_poses = np.asarray(hand_poses, dtype=np.float64)
        starts = np.flatnonzero((lift >= 0.05) & (contacts > 0))
        if not len(starts):
            transport = False
        else:
            start = int(starts[0])
            hand_rotation = Rotation.from_euler("xyz", hand_poses[:, 3:6])
            local_position = hand_rotation.inv().apply(positions - hand_poses[:, :3])
            terminal_position = np.median(local_position[-30:], axis=0)
            translation_change = float(np.linalg.norm(local_position[start:] - terminal_position, axis=1).max())
            local_rotation = hand_rotation.inv() * Rotation.from_quat(np.asarray(object_quaternions))
            rotation_change = float(np.degrees((local_rotation[-1].inv() * local_rotation[start:]).magnitude().max()))
            transport = bool(stable and translation_change <= 0.03 and rotation_change <= 30.0)
    return {
        "reference_isaac_success": bool(reference.any()),
        "reference_isaac_terminal_success": bool(reference[-30:].all()),
        "success": stable,
        "transport_stability_success": transport,
        "max_palm_relative_translation_change_m": translation_change,
        "max_palm_relative_rotation_change_deg": rotation_change,
        "max_lift_m": float(lift.max()), "final_lift_m": float(lift[-1]),
        "terminal_lift_range_m": float(np.ptp(lift[-30:])),
        "contact_steps": int((contacts > 0).sum()),
        "hand_object_contact_steps": int((contacts > 0).sum()),
        "peak_to_final_drop_m": float(lift.max() - lift[-1]),
        "max_xy_drift_m": float(drift.max()),
        "terminal_contact_ratio": float((contacts[-30:] > 0).mean()),
    }


class ReplayBatchEnv:
    """在一个PhysX仿真中并行重放多条候选轨迹。"""

    def __init__(self, cases, steps_per_frame=3, hold_steps=30):
        """输入同手Case列表和执行时序，输出可reset/step的批量环境；每Case对应一个env。"""
        gymapi = _isaac()
        self.cases, self.n, self.hand = cases, len(cases), cases[0].hand
        self.steps_per_frame, self.hold_steps = int(steps_per_frame), int(hold_steps)
        self.horizon = 70 * self.steps_per_frame + self.hold_steps
        self.gym, self.sim = create_sim()
        asset, properties, self.dof_names = load_hand_asset(self.gym, self.sim, self.hand)
        self.lower, self.upper = np.asarray(properties["lower"]), np.asarray(properties["upper"])
        self.envs, self.hands, self.objects, self.open_commands = [], [], [], []
        self.hand_indices, self.object_indices = [], []
        for case in cases:
            env = self.gym.create_env(self.sim, gymapi.Vec3(-1, -1, -0.2), gymapi.Vec3(1, 1, 1), 1)
            hand_actor = self.gym.create_actor(env, asset, gymapi.Transform(), self.hand, 0, 1)
            self.gym.set_actor_dof_properties(env, hand_actor, properties)
            object_asset = load_object_asset(self.gym, self.sim, case.object_dir)
            object_actor = self.gym.create_actor(env, object_asset,
                object_pose(case.object_dir, case.scale, case.rotation), "object", 0, 0)
            self.gym.set_actor_scale(env, object_actor, case.scale)
            shape_properties = self.gym.get_actor_rigid_shape_properties(env, object_actor)
            for item in shape_properties:
                item.friction = 1.0
            self.gym.set_actor_rigid_shape_properties(env, object_actor, shape_properties)
            opened = case.target_frames[0].copy(); opened[6:] = 0.0
            self.envs.append(env); self.hands.append(hand_actor); self.objects.append(object_actor)
            self.open_commands.append(map_command(self.hand, opened, self.dof_names, case.target_metadata))
            self.hand_indices.append(actor_indices(self.gym, env, hand_actor))
            self.object_indices.append(actor_indices(self.gym, env, object_actor))
        self._capture_rest_state()

    def _set_hand(self, index, target):
        """输入环境编号/物理目标，同时写DOF状态和位置目标，用于确定性复位。"""
        gymapi = _isaac()
        state = self.gym.get_actor_dof_states(self.envs[index], self.hands[index], gymapi.STATE_ALL)
        state["pos"], state["vel"] = target, 0.0
        self.gym.set_actor_dof_states(self.envs[index], self.hands[index], state, gymapi.STATE_ALL)
        self.gym.set_actor_dof_position_targets(self.envs[index], self.hands[index], target)

    def _object_state(self, index):
        """输入环境编号，输出物体位置/四元数/速度13维状态，用于记录评估轨迹。"""
        gymapi = _isaac()
        state = self.gym.get_actor_rigid_body_states(self.envs[index], self.objects[index], gymapi.STATE_ALL)
        p, r, v = state["pose"]["p"][0], state["pose"]["r"][0], state["vel"][0]
        return np.asarray([p[0], p[1], p[2], r[0], r[1], r[2], r[3],
                           v["linear"][0], v["linear"][1], v["linear"][2],
                           v["angular"][0], v["angular"][1], v["angular"][2]], dtype=np.float32)

    def _capture_rest_state(self):
        """输入无，保存张手落稳30步后的物体状态，作为每个候选的相同起点。"""
        gymapi = _isaac()
        for i in range(self.n):
            self._set_hand(i, self.open_commands[i])
        for _ in range(30):
            self.gym.simulate(self.sim); self.gym.fetch_results(self.sim, True)
        self.rest_states, initial = [], []
        for i in range(self.n):
            self.rest_states.append(self.gym.get_actor_rigid_body_states(
                self.envs[i], self.objects[i], gymapi.STATE_ALL).copy())
            initial.append(self._object_state(i)[:3])
        self.initial = np.stack(initial)

    def reset(self):
        """输入输出无；将手物恢复到相同起点并清空轨迹记录。"""
        gymapi = _isaac()
        for i in range(self.n):
            self.gym.set_actor_rigid_body_states(self.envs[i], self.objects[i], self.rest_states[i], gymapi.STATE_ALL)
            self._set_hand(i, self.open_commands[i])
        self.positions = [[] for _ in range(self.n)]
        self.contacts = [[] for _ in range(self.n)]
        self.hand_poses = [[] for _ in range(self.n)]
        self.object_quaternions = [[] for _ in range(self.n)]

    def command(self, step):
        """输入物理步，输出当前批次命令；在相邻20 Hz数据帧间做3个60 Hz线性插值。"""
        frame = min(step // self.steps_per_frame, 69)
        target = np.stack([case.target_frames[frame] for case in self.cases])
        if step >= 70 * self.steps_per_frame:
            return target
        previous = np.stack([
            case.target_frames[frame - 1] if frame else np.concatenate([
                case.target_frames[0, :6],
                np.zeros(case.target_frames.shape[1] - 6, dtype=np.float32),
            ])
            for case in self.cases
        ])
        alpha = (step % self.steps_per_frame + 1) / self.steps_per_frame
        return previous * (1.0 - alpha) + target * alpha

    def step(self, step):
        """输入物理步，执行插值命令，输出无；内部记录物体位姿和手物接触。"""
        commands = self.command(step)
        for i in range(self.n):
            physical = map_command(self.hand, commands[i], self.dof_names, self.cases[i].target_metadata)
            self.gym.set_actor_dof_position_targets(self.envs[i], self.hands[i], np.clip(physical, self.lower, self.upper))
        self.gym.simulate(self.sim); self.gym.fetch_results(self.sim, True)
        gymapi = _isaac()
        wrist_names = WRIST_XHAND if self.hand == "xhand" else WRIST_LINKER_WUJI
        for i in range(self.n):
            obj = self._object_state(i)
            contact = pair_contact_count(self.gym.get_env_rigid_contacts(self.envs[i]),
                                         self.hand_indices[i], self.object_indices[i])
            dof = self.gym.get_actor_dof_states(self.envs[i], self.hands[i], gymapi.STATE_POS)["pos"]
            by_name = dict(zip(self.dof_names, dof))
            self.positions[i].append(obj[:3]); self.contacts[i].append(contact)
            self.hand_poses[i].append([by_name[name] for name in wrist_names])
            self.object_quaternions[i].append(obj[3:7])

    def metrics(self):
        """输入无，输出每个环境的参考成功与稳定运输指标。"""
        return [success_metrics(self.positions[i], self.initial[i], self.contacts[i],
                                self.hand_poses[i], self.object_quaternions[i])
                for i in range(self.n)]

    def close(self):
        """输入输出无；销毁sim并释放PhysX资源。"""
        self.gym.destroy_sim(self.sim)


def run_replay(env):
    """输入``ReplayBatchEnv``，输出每条Case指标；执行210个插值步和30个末帧保持步。"""
    env.reset()
    for step in range(env.horizon):
        env.step(step)
    return env.metrics()


def main():
    """输入单物体CLI参数，输出JSON指标；这是最小实现的可执行重放入口。"""
    parser = argparse.ArgumentParser(description="重放一条重定向轨迹")
    parser.add_argument("--hand", choices=("linker", "xhand", "wuji"), required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--object-dir", required=True)
    parser.add_argument("--source-index", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cases = cases_from_files(args.hand, args.source, args.target, args.object_dir, [args.source_index])
    env = ReplayBatchEnv(cases)
    try:
        result = run_replay(env)[0]
    finally:
        env.close()
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
