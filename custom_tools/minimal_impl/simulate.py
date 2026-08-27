"""使用官方Isaac Gym任务进行真实闭环评测或在线数据采集。

本文件不引用原 ``custom_tools`` 的任何模块。它复用仓库官方的 ``dexgrasp`` 环境、
配置和mesh，并用 ``ShadowHandGraspDexRepIjrr2`` 接收内存中的多物体轨迹。
"""

import argparse
import copy
import os
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEXGRASP_ROOT = REPO_ROOT / "dexgrasp"
CATEGORY_NAMES = ("bottle", "mug", "bowl", "camera")


def parse_args():
    """输入终端参数，输出评测或采集配置；作用是提供唯一Isaac Gym运行入口。"""
    parser = argparse.ArgumentParser(description="精简版真实Isaac Gym闭环入口")
    parser.add_argument("mode", choices=("evaluate", "collect"))
    parser.add_argument("--student", required=True)
    parser.add_argument("--teacher", action="append", default=[],
                        help="采集模式需要四次 CATEGORY=CHECKPOINT")
    parser.add_argument("--trajectory-root", required=True)
    parser.add_argument("--object-id", action="append", required=True)
    parser.add_argument("--trajectory-offset", type=int, default=0)
    parser.add_argument("--trajectories-per-object", type=int, default=0)
    parser.add_argument("--policy-steps", type=int, default=70)
    parser.add_argument("--hold-steps", type=int, default=52)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--env-config", default=str(
        DEXGRASP_ROOT / "cfg/shadow_hand_grasp_dexrep_ijrr.yaml"))
    parser.add_argument("--train-config", default=str(
        DEXGRASP_ROOT / "cfg/ppo1/config.yaml"))
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--rl-device", default="cuda:0")
    parser.add_argument("--show-viewer", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_teachers(values):
    """输入四条类别教师声明，输出类别到checkpoint路径的字典。

    内部检查四类是否完整且不重复；作用是保证在线监督的教师路由正确。
    """
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--teacher格式必须是 CATEGORY=CHECKPOINT")
        category, path = value.split("=", 1)
        if category not in CATEGORY_NAMES or category in result:
            raise ValueError(f"教师类别无效或重复：{category}")
        result[category] = str(Path(path).expanduser().resolve())
    if set(result) != set(CATEGORY_NAMES):
        raise ValueError("collect模式必须恰好提供四个类别教师")
    return result


def load_trajectory_data(cli):
    """读取原始轨迹必需的三个字段，并保留原文件中的轨迹编号。"""
    result, source_indices = [], []
    root = Path(cli.trajectory_root).expanduser().resolve()
    for object_id in cli.object_id:
        path = root / f"{object_id}.npy"
        data = np.load(str(path), allow_pickle=True).item()
        start = cli.trajectory_offset
        stop = len(data["grasp_seqs"])
        if cli.trajectories_per_object:
            stop = min(stop, start + cli.trajectories_per_object)
        indices = np.arange(start, stop, dtype=np.int64)
        if not len(indices):
            raise ValueError(f"{object_id}没有选中任何轨迹")
        item = {"obj_code": object_id}
        for key in ("obj_scale", "obj_rotmat", "grasp_seqs"):
            item[key] = data[key][indices].copy()
        result.append(item)
        source_indices.append(indices)
    return result, source_indices


def official_args(cli):
    """调用官方参数解析器，但不让它看到精简版自己的CLI参数。"""
    argv = [
        sys.argv[0], "--task=ShadowHandGraspDexRepIjrr", "--algo=ppo1",
        f"--seed={cli.seed}", f"--rl_device={cli.rl_device}",
        f"--sim_device={cli.sim_device}", "--logdir=logs/dexgrasp_minimal",
        f"--cfg_train={Path(cli.train_config).resolve()}",
        f"--cfg_env={Path(cli.env_config).resolve()}",
    ]
    if not cli.show_viewer:
        argv.append("--headless")
    saved = sys.argv
    try:
        sys.argv = argv
        from utils.config import get_args
        return get_args()
    finally:
        sys.argv = saved


def build_task(cli, trajectory_data):
    """用官方Ijrr2环境创建一批真实物理轨迹。"""
    from utils.config import load_cfg, parse_sim_params, set_seed
    from tasks.shadow_hand_grasp_dexrep_ijrr2 import ShadowHandGraspDexRepIjrr2

    args = official_args(cli)
    cfg, cfg_train, _ = load_cfg(args)
    set_seed(cli.seed, cfg_train.get("torch_deterministic", False))
    cfg = copy.deepcopy(cfg)
    cfg["seed"] = cli.seed
    cfg["env"]["seed"] = cli.seed
    cfg["env"]["env_mode"] = "bc_env_infer"
    cfg["env"]["observationType"] = "DexRep"
    cfg["env"]["obj_type"] = "one"
    cfg["env"]["object_code_dict"] = list(cli.object_id)
    cfg["env"]["numEnvs"] = sum(len(item["grasp_seqs"]) for item in trajectory_data)
    cfg["env"].setdefault("seq_start_rot_uniform", False)
    sim_params = parse_sim_params(args, cfg, cfg_train)
    return ShadowHandGraspDexRepIjrr2(
        cfg=cfg, sim_params=sim_params, physics_engine=args.physics_engine,
        device_type=args.device, device_id=args.device_id, headless=args.headless,
        is_multi_agent=False, npy_list=trajectory_data,
    )


def reset_task(task, torch):
    """输入Isaac任务和torch模块，输出episode首帧观测。

    内部清零进度与成功标志并触发统一重置；作用是建立可重复的闭环起点。
    """
    task.random_time = False
    task.reset_buf[:] = 1
    task.progress_buf[:] = 0
    task.step(torch.zeros((task.num_envs, 28), device=task.device), id=-1)
    task.progress_buf[:] = 0
    task.successes[:] = 0
    return task.obs_buf.clone()


def environment_categories(task, object_ids, torch):
    """输入并行环境的物体索引，输出每个环境的0到3类别张量。

    内部从物体ID解析类别；作用是为每个并行环境生成Task-ID。
    """
    return torch.tensor(
        [CATEGORY_NAMES.index(object_ids[int(index)].split("-", 2)[1])
         for index in task.object_idxs],
        device=task.device, dtype=torch.long,
    )


def evaluate(cli, task, student, torch, helpers):
    """输入运行配置、环境和学生，输出物体级官方成功率字典。

    内部自主决策70步、保持末动作52步并统计峰值成功和抬升；作用是完成真实闭环评测。
    """
    category_one_hot, PolicyRuntime = helpers
    observation = reset_task(task, torch)
    categories = environment_categories(task, cli.object_id, torch)
    task_id = category_one_hot(categories).to(observation)
    runtime = PolicyRuntime(student)
    runtime.reset(observation)
    object_index = torch.tensor(task.object_idxs, device=task.device)
    peak = [0] * len(cli.object_id)
    initial_height = task.object_pos[:, 2].clone()
    maximum_lift = torch.zeros(task.num_envs, device=task.device)
    last_action = None

    for step in range(cli.policy_steps + cli.hold_steps):
        action = (runtime.act(observation, task_id, step)
                  if step < cli.policy_steps else last_action)
        last_action = action
        task.step(action, step + 1)
        observation = task.obs_buf.clone()
        maximum_lift = torch.maximum(
            maximum_lift, task.object_pos[:, 2] - initial_height)
        for index in range(len(cli.object_id)):
            mask = object_index == index
            peak[index] = max(peak[index], int(((task.successes > 0) & mask).sum()))

    rows = []
    for index, object_id in enumerate(cli.object_id):
        mask = object_index == index
        count = int(mask.sum())
        rows.append({
            "object_id": object_id,
            "trajectory_count": count,
            "official_peak_success_count": peak[index],
            "official_peak_success_rate": peak[index] / count,
            "mean_maximum_lift_m": float(maximum_lift[mask].mean()),
        })
    return {
        "seed": cli.seed,
        "checkpoint": str(Path(cli.student).resolve()),
        "macro_official_peak_success_rate": float(np.mean(
            [row["official_peak_success_rate"] for row in rows])),
        "objects": rows,
    }


def collect(cli, task, student, teachers, source_indices, torch, helpers):
    """输入学生、四教师和环境，输出逐帧在线监督数组。

    内部只用学生动作推进物理，教师仅对学生访问状态贴标签；作用是采集DAgger式数据。
    """
    filter_observation, category_one_hot, PolicyRuntime = helpers
    observation = reset_task(task, torch)
    categories = environment_categories(task, cli.object_id, torch)
    task_id = category_one_hot(categories).to(observation)
    runtime = PolicyRuntime(student)
    runtime.reset(observation)
    object_indices = torch.tensor(task.object_idxs, device=task.device)
    local_trajectory = torch.empty(task.num_envs, dtype=torch.long, device=task.device)
    cursor = 0
    for indices in source_indices:
        local_trajectory[cursor:cursor + len(indices)] = torch.from_numpy(indices).to(task.device)
        cursor += len(indices)
    records = {key: [] for key in (
        "observations", "teacher_actions", "student_actions", "category_indices",
        "object_indices", "trajectory_indices", "frame_indices")}

    for frame in range(cli.policy_steps):
        processed = filter_observation(observation)
        student_action = runtime.act(observation, task_id, frame)
        teacher_action = torch.empty_like(student_action)
        for category_index, category in enumerate(CATEGORY_NAMES):
            mask = categories == category_index
            if torch.any(mask):
                teacher_action[mask] = teachers[category](processed[mask]).clamp(-1, 1)
        records["observations"].append(processed.cpu().numpy())
        records["teacher_actions"].append(teacher_action.cpu().numpy())
        records["student_actions"].append(student_action.cpu().numpy())
        records["category_indices"].append(categories.cpu().numpy())
        records["object_indices"].append(object_indices.cpu().numpy())
        records["trajectory_indices"].append(local_trajectory.cpu().numpy())
        records["frame_indices"].append(np.full(task.num_envs, frame, dtype=np.int64))
        task.step(student_action, frame + 1)
        observation = task.obs_buf.clone()
    # 每帧记录为[num_envs,...]，转置后按轨迹连续展开，正好兼容project_data。
    output = {}
    for key, values in records.items():
        array = np.stack(values, axis=1)
        output[key] = array.reshape((-1,) + array.shape[2:])
    return output


def main():
    """输入命令行，输出YAML评测结果或NPZ在线数据。

    内部延迟导入Isaac Gym、创建环境并调用evaluate/collect；作用是串联完整物理流程。
    """
    cli = parse_args()
    required = [cli.student, cli.trajectory_root, cli.env_config, cli.train_config]
    if cli.mode == "collect":
        teacher_paths = parse_teachers(cli.teacher)
        required.extend(teacher_paths.values())
    else:
        teacher_paths = {}
    missing = [path for path in required if not Path(path).expanduser().exists()]
    if missing:
        raise FileNotFoundError(f"缺少输入：{missing}")
    if cli.dry_run:
        print("LEAN_SIMULATION_DRY_RUN=READY")
        print(vars(cli))
        return

    # Isaac Gym必须在torch之前导入；因此所有torch相关模块都延迟到这里。
    sys.path[:0] = [str(REPO_ROOT), str(DEXGRASP_ROOT)]
    import isaacgym  # noqa: F401
    import torch
    from custom_tools.minimal_impl.model import (
        PolicyRuntime, category_one_hot, filter_observation, load_project_checkpoint)

    trajectory_data, source_indices = load_trajectory_data(cli)
    student = load_project_checkpoint(cli.student, device=cli.rl_device)
    teachers = {
        category: load_project_checkpoint(
            path, use_task_id=False, history_steps=0, chunk_horizon=1,
            device=cli.rl_device)
        for category, path in teacher_paths.items()
    }
    task = None
    original_cwd = Path.cwd()
    try:
        os.chdir(str(DEXGRASP_ROOT))
        task = build_task(cli, trajectory_data)
        helpers = (filter_observation, category_one_hot, PolicyRuntime)
        with torch.no_grad():
            result = (
                evaluate(cli, task, student, torch, helpers)
                if cli.mode == "evaluate"
                else collect(cli, task, student, teachers, source_indices, torch, helpers)
            )
        output = Path(cli.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        if cli.mode == "evaluate":
            import yaml
            with output.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(result, handle, allow_unicode=True, sort_keys=False)
        else:
            np.savez_compressed(str(output), **result)
        print(f"LEAN_{cli.mode.upper()}_COMPLETE={output}")
    finally:
        if task is not None:
            task.clean_sim()
        os.chdir(str(original_cwd))


if __name__ == "__main__":
    main()
