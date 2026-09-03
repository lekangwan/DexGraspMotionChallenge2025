"""最终重定向方法的教学版物理在环CEM与Rank-5协同优化。

输入：运动学重定向后的70帧轨迹、物体信息和Isaac批量评估器。
输出：经过Global或Rank-5残差修正、重复确认后的目标轨迹。
逻辑：CEM从高分候选更新高斯分布；候选不稳定时恢复零修改基线。
作用：展示最终方法怎样从“几何姿态相似”进一步优化到“物理上抓得住”。
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .config import HANDS
from .data import Case, load_npy
from .simulate import ReplayBatchEnv, run_replay


def smoothstep(values):
    """输入任意实数进度，输出0到1的平滑门；作用是避免阶段边界动作突变。"""
    values = np.clip(values, 0.0, 1.0)
    return values * values * (3.0 - 2.0 * values)


def phase_frames(frames):
    """输入70帧动作，输出闭合开始和抬升前抓稳帧；依据手指变化与手腕上升自动检测。"""
    fingers = np.asarray(frames[:, 6:], dtype=np.float32)
    movement = np.linalg.norm(fingers - fingers[0], axis=1)
    close_candidates = np.flatnonzero(movement >= max(0.2 * float(movement.max()), 1e-3))
    close = int(close_candidates[0]) if len(close_candidates) else 20
    base_z = float(np.min(frames[close:, 2]))
    lift_candidates = np.flatnonzero(
        (np.arange(len(frames)) > close) & (frames[:, 2] >= base_z + 0.03)
    )
    lift = int(lift_candidates[0]) if len(lift_candidates) else min(55, len(frames) - 1)
    return close, max(close + 1, lift - 1)


def apply_global(frames, parameters):
    """输入基线轨迹与低维全局参数，输出手腕/分指闭合/抬升平滑修正后的轨迹。"""
    result = np.asarray(frames, dtype=np.float32).copy()
    parameters = np.asarray(parameters, dtype=np.float32)
    time = np.linspace(0.0, 1.0, len(result), dtype=np.float32)
    approach = smoothstep((time - 0.15) / 0.35)[:, None]
    closure = smoothstep((time - 0.35) / 0.30)[:, None]
    lift = smoothstep((time - 0.65) / 0.25)
    result[:, :3] += approach * parameters[:3]
    result[:, 3:6] += approach * parameters[3:6]
    offsets = parameters[6:-1]
    for offset, indices in zip(offsets, np.array_split(np.arange(6, result.shape[1]), len(offsets))):
        result[:, indices] += closure * float(offset)
    result[:, 2] += lift * float(parameters[-1])
    return result


def build_synergy_basis(trajectories, rank=5):
    """输入多条同构目标轨迹，输出前rank个归一化关节协同方向；内部用SVD提取共同闭合/抬升模式。"""
    patterns = []
    for frames in np.asarray(trajectories, dtype=np.float32):
        close, grasp = phase_frames(frames)
        for vector in (frames[grasp, 6:] - frames[close, 6:], frames[-1, 6:] - frames[grasp, 6:]):
            norm = float(np.linalg.norm(vector))
            if norm >= 1e-4:
                patterns.append(vector / norm)
    _, _, right = np.linalg.svd(np.stack(patterns), full_matrices=False)
    return right[:min(int(rank), len(right))].astype(np.float32)


def apply_synergy(frames, parameters, basis):
    """输入Rank-5基底与闭合/抬升系数，输出整手协同残差轨迹；避免逐关节独立乱动。"""
    result = np.asarray(frames, dtype=np.float32).copy()
    basis = np.asarray(basis, dtype=np.float32)
    parameters = np.asarray(parameters, dtype=np.float32)
    rank = len(basis)
    close, grasp = phase_frames(result)
    indices = np.arange(len(result), dtype=np.float32)
    close_gate = smoothstep((indices - close) / max(1, grasp - close))
    lift_gate = smoothstep((indices - grasp) / 10.0)
    result[:, 6:] += (
        close_gate[:, None] * (parameters[:rank] @ basis)[None]
        + lift_gate[:, None] * (parameters[rank:] @ basis)[None]
    )
    return result


def physics_score(metric):
    """输入单次物理指标，输出正式CEM分数；奖励抬升/接触/运输，惩罚回落/漂移。"""
    score = 100.0 * float(metric["success"])
    score += 40.0 * float(metric["transport_stability_success"])
    score += 25.0 * np.clip(metric["final_lift_m"] / 0.15, -1.0, 1.2)
    score += 10.0 * np.clip(metric["max_lift_m"] / 0.15, 0.0, 1.2)
    score += 5.0 * float(metric["terminal_contact_ratio"])
    score += 2.0 * np.clip(metric["hand_object_contact_steps"] / 150.0, 0.0, 1.0)
    score -= 10.0 * max(0.0, metric["peak_to_final_drop_m"] - 0.03)
    score -= 5.0 * max(0.0, metric["max_xy_drift_m"] - 0.25)
    translation = metric.get("max_palm_relative_translation_change_m")
    rotation = metric.get("max_palm_relative_rotation_change_deg")
    if translation is not None:
        score -= 20.0 * max(0.0, float(translation) - 0.03) / 0.03
    if rotation is not None:
        score -= 10.0 * max(0.0, float(rotation) - 30.0) / 30.0
    return float(score)


def cem_search(evaluate, dimension, std, bounds, population=8, elite=2, iterations=2, seed=0):
    """输入批量评估回调和参数范围，输出最佳参数、指标和搜索历史；每轮强制包含零修改基线。"""
    rng = np.random.default_rng(seed)
    mean = np.zeros(dimension, dtype=np.float32)
    std, bounds = np.asarray(std, dtype=np.float32), np.asarray(bounds, dtype=np.float32)
    best_parameters = mean.copy()
    best_metric = None
    best_score = -np.inf
    history = []
    for iteration in range(1, iterations + 1):
        candidates = np.clip(rng.normal(mean, std, (population, dimension)), -bounds, bounds).astype(np.float32)
        candidates[0] = 0.0
        metrics = evaluate(candidates)
        scores = np.asarray([physics_score(item) for item in metrics])
        order = np.argsort(scores)[::-1]
        selected = candidates[order[:elite]]
        mean = selected.mean(axis=0)
        std = np.maximum(selected.std(axis=0), bounds * 0.05)
        if scores[order[0]] > best_score:
            best_score = float(scores[order[0]])
            best_parameters = candidates[order[0]].copy()
            best_metric = metrics[order[0]]
        history.append({"iteration": iteration, "best_score": float(scores[order[0]])})
    return best_parameters, best_metric, history


def make_population_evaluator(base_case, target_metadata, transform, basis=None):
    """输入单条Case和轨迹变换，输出批量PhysX评估回调；每个候选对应一个并行环境。"""
    def evaluate(parameters):
        """输入一批低维参数，输出每个候选完整重放后的物理指标。"""
        cases = [Case(
            base_case.hand, base_case.category, base_case.object_name, base_case.source_index,
            transform(base_case.target_frames, value) if basis is None else transform(base_case.target_frames, value, basis),
            base_case.object_dir, base_case.scale, base_case.rotation, target_metadata,
        ) for value in parameters]
        env = ReplayBatchEnv(cases)
        try:
            return run_replay(env)
        finally:
            env.close()
    return evaluate


def robust_confirmation(evaluate, candidate, margin=1.0, repeats=2):
    """输入候选参数，分别重复评估零修改和候选；候选每次稳定运输且平均分领先才接受。"""
    zero = np.zeros_like(candidate)
    baseline = [evaluate(zero[None])[0] for _ in range(repeats)]
    refined = [evaluate(candidate[None])[0] for _ in range(repeats)]
    accepted = bool(
        all(item["success"] and item["transport_stability_success"] for item in refined)
        and np.mean([physics_score(item) for item in refined])
        > np.mean([physics_score(item) for item in baseline]) + float(margin)
    )
    return accepted, baseline, refined


def main():
    """读取单物体目标NPY，执行一次Global或Rank-5 CEM并保存确认后的完整NPY。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand", choices=tuple(HANDS), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--object-dir", type=Path, required=True)
    parser.add_argument("--source-index", type=int, required=True)
    parser.add_argument("--stage", choices=("global", "synergy"), required=True)
    parser.add_argument("--synergy-basis", type=Path,
                        help="Rank-5阶段必须使用校准集冻结的全局基底NPY")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260901)
    args = parser.parse_args()
    source, target = load_npy(args.source), load_npy(args.target)
    rows = np.flatnonzero(np.asarray(target["source_trajectory_indices"]) == args.source_index)
    if len(rows) != 1:
        raise ValueError("source-index必须在目标NPY中唯一存在")
    row = int(rows[0])
    case = Case(
        args.hand, "unknown", args.source.stem, args.source_index,
        np.asarray(target["grasp_seqs"][row], dtype=np.float32), args.object_dir,
        float(source["obj_scale"][args.source_index]), source["obj_rotmat"][args.source_index], target,
    )
    if args.stage == "global":
        groups = 6 if args.hand == "linker" else 5
        finger_std = [0.12] * groups
        finger_bounds = [0.30] * groups
        std = [0.010] * 3 + [0.10] * 3 + finger_std + [0.020]
        bounds = [0.025] * 3 + [0.25] * 3 + finger_bounds + [0.050]
        transform, basis = apply_global, None
    else:
        if args.synergy_basis is None:
            parser.error("synergy阶段需要--synergy-basis，不得用当前测试物体重新拟合")
        basis = np.load(args.synergy_basis).astype(np.float32)
        std = [0.08] * len(basis) + [0.10] * len(basis)
        bounds = [0.20] * len(basis) + [0.25] * len(basis)
        transform = apply_synergy
    evaluate = make_population_evaluator(case, target, transform, basis)
    parameters, metric, history = cem_search(
        evaluate, len(std), std, bounds, population=8, elite=2, iterations=2, seed=args.seed,
    )
    accepted, baseline_repeats, candidate_repeats = robust_confirmation(evaluate, parameters)
    selected = parameters if accepted else np.zeros_like(parameters)
    output = dict(target)
    sequences = np.asarray(target["grasp_seqs"]).copy()
    sequences[row] = transform(case.target_frames, selected) if basis is None else transform(case.target_frames, selected, basis)
    output["grasp_seqs"] = sequences
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, output, allow_pickle=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({
        "stage": args.stage, "accepted": accepted, "parameters": selected.tolist(),
        "search_metric": metric, "history": history,
        "baseline_confirmation": baseline_repeats, "candidate_confirmation": candidate_repeats,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"accepted={accepted} output={args.output}")


if __name__ == "__main__":
    main()
