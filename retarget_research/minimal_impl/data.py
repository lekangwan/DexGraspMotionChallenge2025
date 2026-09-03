"""读取源轨迹、manifest、目标轨迹和物体形状。

输入：GraspM3 NPY、冻结manifest、重定向后的NPY和COACD网格。
输出：统一的 ``Case`` 对象及NumPy数组。
逻辑：只保留重定向和物理验证真正需要的字段。
作用：把磁盘格式与算法隔开，使后续文件只处理清楚的内存对象。
"""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass
class Case:
    """一条可运行任务。

    输入：手名、类别、物体名、源索引、70帧目标动作和物体几何信息。
    输出：训练环境或评测环境直接消费的数据对象。
    逻辑：把原来分散在manifest、源NPY和目标NPY的信息汇总。
    作用：一条Case对应一个Isaac Gym环境。
    """

    hand: str
    category: str
    object_name: str
    source_index: int
    target_frames: np.ndarray
    object_dir: Path
    scale: float
    rotation: np.ndarray
    target_metadata: dict


def load_npy(path):
    """输入NPY路径，输出其中的字典；逻辑是启用pickle读取GraspM3格式。"""
    return np.load(Path(path), allow_pickle=True).item()


def save_candidate(path, frames, source, indices, metadata=None):
    """保存重定向轨迹。

    输入：输出路径、``(N,70,D)``动作、源字典、源索引和可选元数据。
    输出：一个可被仿真直接读取的NPY文件。
    逻辑：同时复制对应的物体旋转和缩放，避免轨迹与物体错位。
    作用：统一三种手的重定向输出格式。
    """
    indices = np.asarray(indices, dtype=np.int64)
    result = {
        "grasp_seqs": np.asarray(frames, dtype=np.float32),
        "source_trajectory_indices": indices,
        "obj_rotmat": np.asarray(source["obj_rotmat"])[indices],
        "obj_scale": np.asarray(source["obj_scale"])[indices],
    }
    result.update(metadata or {})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, result, allow_pickle=True)


def cases_from_files(hand, source_path, target_path, object_dir, indices=None, category="unknown"):
    """从一个物体文件建立若干Case。

    输入：手名、源/目标NPY、物体目录和可选源轨迹索引。
    输出：按目标文件顺序排列的Case列表。
    逻辑：用 ``source_trajectory_indices`` 找回每条目标轨迹的物体姿态。
    作用：为单物体重放和快速训练提供最简单入口。
    """
    source = load_npy(source_path)
    target = load_npy(target_path)
    source_indices = np.asarray(target["source_trajectory_indices"], dtype=np.int64)
    wanted = set(source_indices.tolist() if indices is None else map(int, indices))
    result = []
    for row, source_index in enumerate(source_indices):
        if int(source_index) not in wanted:
            continue
        result.append(Case(
            hand, category, Path(source_path).stem, int(source_index),
            np.asarray(target["grasp_seqs"][row], dtype=np.float32), Path(object_dir),
            float(np.asarray(source["obj_scale"])[source_index]),
            np.asarray(source["obj_rotmat"][source_index], dtype=np.float32), target,
        ))
    return result


def cases_from_manifest(hand, manifest_path, target_dir, limit=None):
    """从正式manifest建立多物体Case。

    输入：手、manifest、目标目录和可选数量上限。
    输出：可并行创建Isaac环境的Case列表。
    逻辑：按物体名读取一次源/目标文件，再按源索引找到目标行。
    作用：为正式manifest批量重放建立Case。
    """
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    cases = []
    for entry in manifest["entries"]:
        source = load_npy(entry["source_path"])
        target = load_npy(Path(target_dir) / f"{entry['object_name']}.npy")
        rows = {int(value): row for row, value in enumerate(target["source_trajectory_indices"])}
        for source_index in entry["trajectory_indices"]:
            cases.append(Case(
                hand, entry.get("category", "unknown"), entry["object_name"], int(source_index),
                np.asarray(target["grasp_seqs"][rows[int(source_index)]], dtype=np.float32),
                Path(entry["object_asset_path"]),
                float(np.asarray(source["obj_scale"])[source_index]),
                np.asarray(source["obj_rotmat"])[source_index], target,
            ))
            if limit and len(cases) >= limit:
                return cases
    return cases
