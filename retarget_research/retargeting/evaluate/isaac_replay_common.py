"""提供不同目标手共享的Isaac Gym CPU物理重放基础函数。

输入：仿真参数、数据集物体资产、目标DOF轨迹和成功阈值。
输出：统一的CPU PhysX场景、物体位置/接触曲线和成功统计。
内部逻辑：固定物体初始化、20 Hz插值执行与持续抬升判据，仅把手资产留给适配器。
作用：保证XHand、Linker和后续Wuji使用同一物理口径，而非各写一套评估规则。
"""

import json
from pathlib import Path

from isaacgym import gymapi
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

try:
    from .contact_sample_utils import collect_hand_object_local_contacts
except ImportError:
    from contact_sample_utils import collect_hand_object_local_contacts
from tracking_metrics import summarize_dof_tracking


class IsaacCameraRecorder:
    """把少量选定轨迹的Isaac相机画面直接写成MP4。"""

    def __init__(
        self,
        gym,
        sim,
        env,
        output_path,
        width=640,
        height=480,
        fps=20,
        capture_every=3,
    ):
        """创建固定视角相机和imageio视频写入器。

        输入：Gym场景、MP4路径、分辨率、输出帧率和物理步抽帧间隔。
        输出：可调用`capture/close`的录像器。
        内部逻辑：默认60 Hz物理每3步取一帧，得到与源轨迹20 Hz一致的视频。
        作用：只为自动选出的成功/失败案例按需重跑录像，避免1000条全录造成巨量I/O。
        """
        import imageio.v2 as imageio

        self.gym = gym
        self.sim = sim
        self.env = env
        self.width = int(width)
        self.height = int(height)
        self.capture_every = int(capture_every)
        if self.capture_every <= 0:
            raise ValueError("视频抽帧间隔必须为正数")
        properties = gymapi.CameraProperties()
        properties.width = self.width
        properties.height = self.height
        properties.horizontal_fov = 50.0
        properties.enable_tensors = False
        self.camera = gym.create_camera_sensor(env, properties)
        if int(self.camera) < 0:
            raise RuntimeError(
                "Isaac相机创建失败；录像需要可用的graphics device，"
                "请在带正确NVIDIA驱动的终端运行，普通无视频评测不受影响"
            )
        gym.set_camera_location(
            self.camera,
            env,
            gymapi.Vec3(0.38, -0.38, 0.30),
            gymapi.Vec3(0.0, 0.0, 0.16),
        )
        self.output = Path(output_path).resolve()
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.writer = imageio.get_writer(
            self.output, fps=int(fps), codec="libx264", quality=8
        )
        self.frame_count = 0

    def capture(self, physics_step):
        """在满足抽帧间隔时渲染当前场景并追加一帧RGB。"""
        if int(physics_step) % self.capture_every:
            return
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        image = self.gym.get_camera_image(
            self.sim, self.env, self.camera, gymapi.IMAGE_COLOR
        )
        rgba = np.asarray(image, dtype=np.uint8).reshape(
            self.height, self.width, 4
        )
        self.writer.append_data(rgba[:, :, :3])
        self.frame_count += 1

    def close(self):
        """关闭编码器并返回绝对视频路径和实际帧数。"""
        if self.writer is not None:
            self.writer.close()
            self.writer = None
        return {"video": str(self.output), "video_frame_count": self.frame_count}


def create_cpu_sim(dt, substeps, enable_graphics=False):
    """创建无GPU流水线的PhysX仿真并添加Z轴地面。

    输入：外层物理步长、每步PhysX子步数和是否为少量案例启用图形设备。
    输出：Isaac Gym接口对象和sim句柄。
    逻辑：显式关闭GPU PhysX/pipeline，设置标准重力和统一求解器参数。
    作用：让三只目标手在当前无CUDA驱动机器上仍能公平做接触验证。
    """
    gym = gymapi.acquire_gym()
    params = gymapi.SimParams()
    params.dt = float(dt)
    params.substeps = int(substeps)
    params.up_axis = gymapi.UP_AXIS_Z
    params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    params.use_gpu_pipeline = False
    params.physx.use_gpu = False
    params.physx.solver_type = 1
    params.physx.num_position_iterations = 8
    params.physx.num_velocity_iterations = 1
    params.physx.contact_offset = 0.002
    params.physx.rest_offset = 0.0
    graphics_device = 0 if enable_graphics else -1
    sim = gym.create_sim(0, graphics_device, gymapi.SIM_PHYSX, params)
    if sim is None:
        raise RuntimeError("Isaac Gym CPU PhysX创建失败")
    plane = gymapi.PlaneParams()
    plane.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    gym.add_ground(sim, plane)
    return gym, sim


def object_start_pose(object_dir, scale, rotation_matrix, clearance):
    """根据物体mesh计算与地面无穿透的初始位姿。

    输入：物体目录、缩放、3×3旋转矩阵和离地间隙。
    输出：Isaac Transform及旋转缩放后mesh的最低Z值。
    逻辑：变换`decomposed.obj`顶点，使最低点最终位于Z=`clearance`。
    作用：复现官方环境根据实例方向自动把物体摆到桌面的初始化原则。
    """
    mesh = trimesh.load_mesh(object_dir / "coacd" / "decomposed.obj", process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    transformed = vertices @ np.asarray(rotation_matrix, dtype=np.float64).T * float(scale)
    min_z = float(transformed[:, 2].min())
    quaternion = Rotation.from_matrix(rotation_matrix).as_quat()
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0.0, -min_z + float(clearance))
    pose.r = gymapi.Quat(*[float(value) for value in quaternion])
    return pose, min_z


def load_object_asset(gym, sim, object_dir):
    """加载数据集提供的COACD动态物体。

    输入：Gym接口、sim句柄和单个物体目录。
    输出：可受重力和接触作用的object asset。
    逻辑：加载`coacd_1.urdf`，使用统一密度并让PhysX重算质心和惯量。
    作用：所有目标手使用相同真实碰撞几何，而不是不同的简化形状。
    """
    options = gymapi.AssetOptions()
    options.density = 100.0
    options.fix_base_link = False
    options.disable_gravity = False
    options.override_com = True
    options.override_inertia = True
    options.use_mesh_materials = True
    options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
    asset = gym.load_asset(sim, str(object_dir / "coacd"), "coacd_1.urdf", options)
    if asset is None:
        raise RuntimeError(f"物体资产加载失败: {object_dir}")
    return asset


def set_dof_state_and_target(gym, env, actor, target):
    """同时设置手的DOF状态和位置控制目标。

    输入：Gym接口、环境、手actor和完整DOF目标。
    输出：无返回值；当前位置等于目标且速度清零。
    逻辑：先写状态，再写position target，消除初始化追踪误差。
    作用：为不同目标手建立同样确定性的轨迹起点。
    """
    states = gym.get_actor_dof_states(env, actor, gymapi.STATE_ALL)
    states["pos"] = target
    states["vel"] = 0.0
    gym.set_actor_dof_states(env, actor, states, gymapi.STATE_ALL)
    gym.set_actor_dof_position_targets(env, actor, target)


def read_object_position(gym, env, actor):
    """读取动态物体根刚体的世界坐标。

    输入：Gym接口、环境和物体actor。
    输出：`[x,y,z]`三维NumPy数组。
    逻辑：取得actor第一个刚体pose的位置字段。
    作用：提供抬升高度、平面漂移和最终位置的原始物理测量。
    """
    states = gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_POS)
    position = states["pose"]["p"][0]
    return np.asarray([position[0], position[1], position[2]], dtype=np.float64)


def read_object_state(gym, env, actor):
    """读取动态物体完整根状态。

    输入：Gym接口、环境和物体actor。
    输出：位置3、四元数4、线速度3、角速度3组成的float32字典。
    内部逻辑：读取第一个刚体的`STATE_ALL`结构并显式复制各字段。
    作用：为进阶策略保存可在闭环部署时重新计算的物体状态观测。
    """
    states = gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_ALL)
    pose = states["pose"][0]
    velocity = states["vel"][0]
    return {
        "object_position": np.asarray(
            [pose["p"][0], pose["p"][1], pose["p"][2]], dtype=np.float32
        ),
        "object_quaternion_xyzw": np.asarray(
            [pose["r"][0], pose["r"][1], pose["r"][2], pose["r"][3]],
            dtype=np.float32,
        ),
        "object_linear_velocity": np.asarray(
            [velocity["linear"][0], velocity["linear"][1], velocity["linear"][2]],
            dtype=np.float32,
        ),
        "object_angular_velocity": np.asarray(
            [velocity["angular"][0], velocity["angular"][1], velocity["angular"][2]],
            dtype=np.float32,
        ),
    }


def append_policy_trace_step(
    trace_sink,
    dof_states,
    commanded_dofs,
    policy_action,
    object_state,
    contact_count,
    source_frame_index,
    interpolation_alpha,
    is_hold,
):
    """向策略专家轨迹追加一个物理步。

    输入：可变字典、手DOF状态、物理/策略命令、物体状态、接触和时序标签。
    输出：无返回值；在字典中的各字段列表末尾追加一项。
    内部逻辑：所有数组立即复制为float32，标量统一为确定类型。
    作用：让三只手共享完全相同的训练轨迹schema，避免各自保存不同观测。
    """
    fields = {
        "hand_dof_position": np.asarray(dof_states["pos"], dtype=np.float32),
        "hand_dof_velocity": np.asarray(dof_states["vel"], dtype=np.float32),
        "commanded_physics_dof_position": np.asarray(
            commanded_dofs, dtype=np.float32
        ),
        "policy_action": np.asarray(policy_action, dtype=np.float32),
        **object_state,
    }
    for name, value in fields.items():
        trace_sink.setdefault(name, []).append(value.copy())
    trace_sink.setdefault("hand_object_contact_count", []).append(int(contact_count))
    trace_sink.setdefault("source_frame_index", []).append(int(source_frame_index))
    trace_sink.setdefault("interpolation_alpha", []).append(
        float(interpolation_alpha)
    )
    trace_sink.setdefault("is_hold", []).append(bool(is_hold))


def read_policy_pre_action_state(
    gym,
    env,
    hand_actor,
    object_actor,
    hand_index_to_name,
    object_indices,
):
    """读取一条控制命令执行前的完整策略状态。

    输入：Gym场景、手/物体actor以及双方刚体索引。
    输出：手DOF状态、物体根状态和当前手物接触点数。
    内部逻辑：在写入新position target和推进PhysX之前读取三部分状态。
    作用：保证监督样本严格表示`当前状态 -> 下一动作`，杜绝使用动作后状态的未来泄漏。
    """
    dof_states = gym.get_actor_dof_states(env, hand_actor, gymapi.STATE_ALL)
    object_state = read_object_state(gym, env, object_actor)
    contacts = gym.get_env_rigid_contacts(env)
    grouped = count_contacts_by_hand_body(
        contacts, hand_index_to_name, object_indices
    )
    return dof_states, object_state, sum(grouped.values())


def finalize_policy_trace(trace_sink):
    """把逐步列表转换成可压缩保存的对齐数组。

    输入：`append_policy_trace_step`累计的字段列表字典。
    输出：每个字段第一维完全一致的NumPy数组字典。
    内部逻辑：数值向量用`stack`，标量列表用`asarray`，最后核对长度。
    作用：在写盘前阻止观测、动作和时序错位进入训练数据。
    """
    vector_fields = {
        "hand_dof_position",
        "hand_dof_velocity",
        "commanded_physics_dof_position",
        "policy_action",
        "object_position",
        "object_quaternion_xyzw",
        "object_linear_velocity",
        "object_angular_velocity",
    }
    arrays = {
        name: np.stack(values) if name in vector_fields else np.asarray(values)
        for name, values in trace_sink.items()
    }
    lengths = {len(value) for value in arrays.values()}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) <= 0:
        raise ValueError(f"策略轨迹字段未对齐: {sorted(lengths)}")
    return arrays


def save_policy_trace(path, trace_sink, metadata):
    """保存单条可训练的物理专家轨迹。

    输入：NPZ路径、逐步轨迹字典和可JSON序列化元数据。
    输出：已解析绝对路径。
    内部逻辑：先完成对齐检查，再以压缩NPZ保存数组和`metadata_json`。
    作用：把重定向评测同时转化为BC、Temporal与Diffusion可复用的专家数据源。
    """
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays = finalize_policy_trace(trace_sink)
    np.savez_compressed(
        output,
        **arrays,
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    return output


def actor_body_indices(gym, env, actor):
    """取得一个actor全部刚体在环境域中的索引。

    输入：Gym接口、环境和actor句柄。
    输出：环境域刚体索引列表。
    逻辑：遍历actor局部刚体编号并转换到`DOMAIN_ENV`。
    作用：后续只统计手和物体之间的接触，排除物体—地面接触。
    """
    return [
        gym.get_actor_rigid_body_index(env, actor, index, gymapi.DOMAIN_ENV)
        for index in range(gym.get_actor_rigid_body_count(env, actor))
    ]


def count_actor_pair_contacts(contacts, first_body_indices, second_body_indices):
    """统计当前物理步中两组刚体之间的接触点数。

    输入：Isaac接触记录和两个actor的环境域刚体索引。
    输出：两组刚体间接触记录数量。
    逻辑：允许body0/body1方向互换，再判断是否分别属于两个集合。
    作用：区分未接触、接触后滑落和稳定包覆三类现象。
    """
    first, second = set(first_body_indices), set(second_body_indices)
    count = 0
    for contact in contacts:
        body0, body1 = int(contact["body0"]), int(contact["body1"])
        if (body0 in first and body1 in second) or (
            body1 in first and body0 in second
        ):
            count += 1
    return count


def count_contacts_by_hand_body(contacts, hand_index_to_name, object_body_indices):
    """按手部刚体名称统计当前步与物体之间的接触点数。

    输入：Isaac接触记录、手环境索引到名称字典和物体刚体索引。
    输出：每个手刚体在当前步的手物接触记录数量字典。
    内部逻辑：兼容body0/body1方向互换，只累计另一端属于物体的手部刚体。
    作用：区分“总接触存在”与“拇指/具体手指真正参与接触”。
    """
    object_indices = set(object_body_indices)
    counts = {name: 0 for name in hand_index_to_name.values()}
    for contact in contacts:
        body0, body1 = int(contact["body0"]), int(contact["body1"])
        if body0 in hand_index_to_name and body1 in object_indices:
            counts[hand_index_to_name[body0]] += 1
        elif body1 in hand_index_to_name and body0 in object_indices:
            counts[hand_index_to_name[body1]] += 1
    return counts


def longest_true_run(mask):
    """统计一维布尔序列最长连续真值长度。

    输入：一维布尔数组。
    输出：最长连续True步数。
    逻辑：单次扫描维护当前长度和历史最大值。
    作用：避免把多个分散的瞬时抬升误加成持续成功。
    """
    best = current = 0
    for value in np.asarray(mask, dtype=bool):
        current = current + 1 if value else 0
        best = max(best, current)
    return int(best)


def replay_position_trajectory(
    gym,
    sim,
    env,
    hand_actor,
    object_actor,
    trajectory,
    open_first,
    settle_steps,
    steps_per_frame,
    hold_steps,
    contact_sample_sink=None,
    trace_sink=None,
    policy_trajectory=None,
    policy_open_first=None,
    video_recorder=None,
):
    """按统一时间插值执行一条完整DOF位置目标轨迹。

    输入：场景句柄、完整DOF轨迹、张开初态、三段物理步数及可选策略命令轨迹。
    输出：物体位置、接触，以及每步手部命令/实际DOF位置。
    逻辑：先张开落稳，再逐帧线性插值，最后保持末帧继续观察。
    作用：把与具体手无关的执行时序固定下来，保证跨手比较公平。

    可选`contact_sample_sink`保存局部接触点；可选`trace_sink`保存训练观测。
    `policy_trajectory`可与物理DOF维度不同；`video_recorder`只在选定案例中启用。
    """
    hand_indices = actor_body_indices(gym, env, hand_actor)
    object_indices = actor_body_indices(gym, env, object_actor)
    hand_names = list(gym.get_actor_rigid_body_names(env, hand_actor))
    hand_index_to_name = dict(zip(hand_indices, hand_names))
    set_dof_state_and_target(gym, env, hand_actor, open_first)
    policy_trajectory = (
        np.asarray(trajectory, dtype=np.float32)
        if policy_trajectory is None
        else np.asarray(policy_trajectory, dtype=np.float32)
    )
    policy_open_first = (
        np.asarray(open_first, dtype=np.float32)
        if policy_open_first is None
        else np.asarray(policy_open_first, dtype=np.float32)
    )
    if len(policy_trajectory) != len(trajectory):
        raise ValueError("策略命令轨迹与物理DOF轨迹帧数不一致")
    for _ in range(settle_steps):
        gym.simulate(sim)
        gym.fetch_results(sim, True)
    initial_position = read_object_position(gym, env, object_actor)

    positions, contact_counts = [], []
    body_contact_counts = {name: [] for name in hand_names}
    actual_dof_positions, commanded_dof_positions = [], []
    previous = open_first
    previous_policy = policy_open_first
    physics_step = 0
    for frame_index, target in enumerate(trajectory):
        for substep in range(1, steps_per_frame + 1):
            alpha = substep / steps_per_frame
            interpolated = previous * (1.0 - alpha) + target * alpha
            policy_target = policy_trajectory[frame_index]
            interpolated_policy = (
                previous_policy * (1.0 - alpha) + policy_target * alpha
            )
            if trace_sink is not None:
                pre_dof_states, pre_object_state, pre_contact_count = (
                    read_policy_pre_action_state(
                        gym,
                        env,
                        hand_actor,
                        object_actor,
                        hand_index_to_name,
                        object_indices,
                    )
                )
            gym.set_actor_dof_position_targets(env, hand_actor, interpolated)
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            if video_recorder is not None:
                video_recorder.capture(physics_step)
            object_state = read_object_state(gym, env, object_actor)
            positions.append(object_state["object_position"].copy())
            dof_states = gym.get_actor_dof_states(
                env, hand_actor, gymapi.STATE_ALL
            )
            actual_dof_positions.append(dof_states["pos"].copy())
            commanded_dof_positions.append(np.asarray(interpolated).copy())
            contacts = gym.get_env_rigid_contacts(env)
            if contact_sample_sink is not None:
                contact_sample_sink.extend(
                    collect_hand_object_local_contacts(
                        contacts,
                        hand_index_to_name,
                        object_indices,
                        physics_step,
                        frame_index,
                    )
                )
            grouped = count_contacts_by_hand_body(
                contacts, hand_index_to_name, object_indices
            )
            contact_count = sum(grouped.values())
            contact_counts.append(contact_count)
            if trace_sink is not None:
                append_policy_trace_step(
                    trace_sink,
                    pre_dof_states,
                    interpolated,
                    interpolated_policy,
                    pre_object_state,
                    pre_contact_count,
                    frame_index,
                    alpha,
                    False,
                )
            for name in hand_names:
                body_contact_counts[name].append(grouped[name])
            physics_step += 1
        previous = target
        previous_policy = policy_target
    for _ in range(hold_steps):
        if trace_sink is not None:
            pre_dof_states, pre_object_state, pre_contact_count = (
                read_policy_pre_action_state(
                    gym,
                    env,
                    hand_actor,
                    object_actor,
                    hand_index_to_name,
                    object_indices,
                )
            )
        gym.set_actor_dof_position_targets(env, hand_actor, trajectory[-1])
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        if video_recorder is not None:
            video_recorder.capture(physics_step)
        object_state = read_object_state(gym, env, object_actor)
        positions.append(object_state["object_position"].copy())
        dof_states = gym.get_actor_dof_states(env, hand_actor, gymapi.STATE_ALL)
        actual_dof_positions.append(dof_states["pos"].copy())
        commanded_dof_positions.append(np.asarray(trajectory[-1]).copy())
        contacts = gym.get_env_rigid_contacts(env)
        if contact_sample_sink is not None:
            contact_sample_sink.extend(
                collect_hand_object_local_contacts(
                    contacts,
                    hand_index_to_name,
                    object_indices,
                    physics_step,
                    len(trajectory),
                )
            )
        grouped = count_contacts_by_hand_body(
            contacts, hand_index_to_name, object_indices
        )
        contact_count = sum(grouped.values())
        contact_counts.append(contact_count)
        if trace_sink is not None:
            append_policy_trace_step(
                trace_sink,
                pre_dof_states,
                trajectory[-1],
                policy_trajectory[-1],
                pre_object_state,
                pre_contact_count,
                len(trajectory),
                1.0,
                True,
            )
        for name in hand_names:
            body_contact_counts[name].append(grouped[name])
        physics_step += 1
    return (
        initial_position,
        np.stack(positions),
        np.asarray(contact_counts, dtype=np.int64),
        {
            name: np.asarray(values, dtype=np.int64)
            for name, values in body_contact_counts.items()
        },
        np.stack(actual_dof_positions),
        np.stack(commanded_dof_positions),
    )


def summarize_body_contacts(body_contact_counts):
    """把逐步刚体接触曲线压缩为易读诊断摘要。

    输入：刚体名称到一维逐步接触数数组的字典。
    输出：有接触步数和单步最大接触点数两个按名称字典。
    内部逻辑：分别统计`count>0`数量和数组最大值，删除全程为零的刚体。
    作用：控制JSON体积，同时明确哪些指节真正参与抓取。
    """
    active = {
        name: np.asarray(values)
        for name, values in body_contact_counts.items()
        if np.any(np.asarray(values) > 0)
    }
    return {
        "hand_body_contact_steps": {
            name: int(np.sum(values > 0)) for name, values in active.items()
        },
        "hand_body_max_simultaneous_contacts": {
            name: int(values.max()) for name, values in active.items()
        },
    }


def compute_success_metrics(
    positions,
    initial_position,
    contact_counts,
    dt,
    lift_threshold,
    max_xy_drift,
    sustain_steps,
    terminal_hold_steps=None,
    max_peak_to_final_drop=0.03,
    max_terminal_lift_range=0.01,
    min_terminal_contact_ratio=1.0,
):
    """从统一物理曲线计算终态稳定抓取与诊断指标。

    输入：位置/接触曲线、初始位置、时间步、抬升/漂移阈值，
    以及末段保持长度、允许回落、末段波动和接触比例。
    输出：保留旧“曾持续抬起”标志，并增加末段高度、回落、波动、
    接触和新的稳定成功标志。
    逻辑：新成功必须曾达到抬升要求，且最后整个保持窗都在阈值上、
    未明显从峰值掉落、窗内不继续下滑，并保持手物接触。
    作用：排除“中途越过10 cm后掉落”的伪成功，统一三手新评测口径。
    """
    positions = np.asarray(positions, dtype=np.float64)
    contact_counts = np.asarray(contact_counts)
    if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) == 0:
        raise ValueError(f"positions必须为非空(T,3)，实际{positions.shape}")
    if contact_counts.shape != (len(positions),):
        raise ValueError("接触曲线必须与位置曲线等长")
    terminal_steps = int(
        sustain_steps if terminal_hold_steps is None else terminal_hold_steps
    )
    if not 1 <= terminal_steps <= len(positions):
        raise ValueError("末段保持步数必须在1到曲线长度之间")
    if max_peak_to_final_drop < 0 or max_terminal_lift_range < 0:
        raise ValueError("允许回落和末段波动不能为负")
    if not 0 <= min_terminal_contact_ratio <= 1:
        raise ValueError("末段接触比例必须在[0,1]")
    lift = positions[:, 2] - initial_position[2]
    xy_drift = np.linalg.norm(
        positions[:, :2] - initial_position[None, :2], axis=1
    )
    valid = (lift >= lift_threshold) & (xy_drift <= max_xy_drift)
    sustained_steps = longest_true_run(valid)
    terminal_lift = lift[-terminal_steps:]
    terminal_valid = valid[-terminal_steps:]
    terminal_contacts = contact_counts[-terminal_steps:] > 0
    legacy_success = bool(sustained_steps >= sustain_steps)
    peak_to_final_drop = float(lift.max() - lift[-1])
    terminal_lift_range = float(np.ptp(terminal_lift))
    terminal_contact_ratio = float(terminal_contacts.mean())
    terminal_stable = bool(
        terminal_valid.all()
        and peak_to_final_drop <= float(max_peak_to_final_drop)
        and terminal_lift_range <= float(max_terminal_lift_range)
        and terminal_contact_ratio >= float(min_terminal_contact_ratio)
    )
    return {
        "initial_object_position_m": initial_position.tolist(),
        "max_lift_m": float(lift.max()),
        "final_lift_m": float(lift[-1]),
        "max_xy_drift_m": float(xy_drift.max()),
        "lift_threshold_m": float(lift_threshold),
        "max_allowed_xy_drift_m": float(max_xy_drift),
        "required_sustain_steps": int(sustain_steps),
        "required_sustain_time_s": float(sustain_steps * dt),
        "longest_sustained_lift_steps": sustained_steps,
        "longest_sustained_lift_time_s": float(sustained_steps * dt),
        "hand_object_contact_steps": int((contact_counts > 0).sum()),
        "max_simultaneous_hand_object_contacts": int(contact_counts.max()),
        "legacy_sustained_success": legacy_success,
        "success_protocol": "stable_30cm_terminal_hold_v2",
        "terminal_hold_steps": terminal_steps,
        "terminal_hold_time_s": float(terminal_steps * dt),
        "terminal_min_lift_m": float(terminal_lift.min()),
        "terminal_max_lift_m": float(terminal_lift.max()),
        "terminal_lift_range_m": terminal_lift_range,
        "peak_to_final_drop_m": peak_to_final_drop,
        "terminal_contact_steps": int(terminal_contacts.sum()),
        "terminal_contact_ratio": terminal_contact_ratio,
        "max_allowed_peak_to_final_drop_m": float(max_peak_to_final_drop),
        "max_allowed_terminal_lift_range_m": float(max_terminal_lift_range),
        "min_required_terminal_contact_ratio": float(min_terminal_contact_ratio),
        "terminal_stable": terminal_stable,
        "success": bool(legacy_success and terminal_stable),
        "object_positions_m": positions.tolist(),
        "hand_object_contact_count_per_step": contact_counts.tolist(),
    }
