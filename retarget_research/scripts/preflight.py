#!/usr/bin/env python3
"""正式运行前只读检查：环境、三手URDF、轨迹字段和物体资产。

输入：参考仓库、完整轨迹目录、物体资产目录和可选CUDA要求。
输出：JSON审计和`RETARGET_PREFLIGHT=PASS/FAIL`。
内部逻辑：检查Python模块、Isaac Gym导入、URDF主动自由度、至少100个标准轨迹文件
及至少100个物体资产；CPU PhysX是基本任务合法路径，CUDA只在显式要求时作为门槛。
作用：把旧的“无CUDA即失败”误判修正为与当前统一CPU评测协议一致的正式门禁。
"""

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET


HANDS = {
    "linkerhand_o6": "scripts/assets/linkerhand/o6/right/linkerhand_o6_right.urdf",
    "xhand": "scripts/assets/xhand_right/urdf/xhand_right.urdf",
    "wujihand": "scripts/assets/wujihand_urdf/urdf/right.urdf",
}


def urdf_info(path: Path):
    """解析URDF的主动/mimic关节数量和名称。"""
    root = ET.parse(path).getroot()
    joints = [j for j in root.findall(".//joint") if j.get("type") != "fixed"]
    mimic = [j for j in joints if j.find("mimic") is not None]
    return {
        "path": str(path.resolve()),
        "nonfixed_joints": len(joints),
        "mimic_joints": len(mimic),
        "active_joints": len(joints) - len(mimic),
        "joint_names": [j.get("name") for j in joints],
    }


def module_status(name):
    """返回当前解释器能否发现指定Python模块。"""
    return importlib.util.find_spec(name) is not None


def isaac_cuda_status():
    """在独立子进程检查Isaac Gym导入和PyTorch CUDA可见性。"""
    code = (
        "import json; import isaacgym; from isaacgym import gymapi; "
        "import torch; print(json.dumps({'isaacgym': True, "
        "'torch': torch.__version__, 'cuda': torch.cuda.is_available(), "
        "'device_count': torch.cuda.device_count()}))"
    )
    run = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
    )
    lines = [line for line in run.stdout.splitlines() if line.startswith("{")]
    if run.returncode == 0 and lines:
        return json.loads(lines[-1])
    return {"isaacgym": False, "error": (run.stderr or run.stdout)[-1000:]}


def inspect_dataset(root: Path, limit=0):
    """检查轨迹目录中全部或指定上限的GraspM3文件。

    输入：目录和检查上限；0表示全部。
    输出：文件数、轨迹总数和错误列表。
    内部逻辑：逐文件核对三字段与`(N,70,28)`，不修改数据。
    作用：正式100物体不再只抽前200个文件造成隐藏坏文件。
    """
    files = sorted(root.glob("*.npy")) if root.exists() else []
    result = {"root": str(root.resolve()), "file_count": len(files), "invalid": []}
    if not files:
        return result
    import numpy as np

    trajectory_total = 0
    inspected = files if limit <= 0 else files[:limit]
    for path in inspected:
        try:
            data = np.load(str(path), allow_pickle=True).item()
            missing = [k for k in ("grasp_seqs", "obj_rotmat", "obj_scale") if k not in data]
            if missing:
                result["invalid"].append({"file": path.name, "missing": missing})
                continue
            seq = data["grasp_seqs"]
            if seq.ndim != 3 or seq.shape[1:] != (70, 28):
                result["invalid"].append({"file": path.name, "shape": list(seq.shape)})
            trajectory_total += int(seq.shape[0])
        except Exception as exc:
            result["invalid"].append({"file": path.name, "error": str(exc)})
    result["inspected_files"] = len(inspected)
    result["inspected_trajectory_total"] = trajectory_total
    return result


def main():
    """执行预检、写报告并用退出码表达是否就绪。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--asset-root", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--dataset-limit", type=int, default=0)
    args = parser.parse_args()

    reference = Path(args.reference_root).expanduser()
    result = {
        "python": sys.version,
        "executable": sys.executable,
        "modules": {name: module_status(name) for name in (
            "numpy", "torch", "nlopt", "autograd", "trimesh",
            "transforms3d", "pytorch_kinematics", "pytorch3d", "torchsdf")},
        "isaac_cuda": isaac_cuda_status(),
        "hands": {},
        "dataset": inspect_dataset(
            Path(args.dataset_root).expanduser(), args.dataset_limit
        ),
        "asset_object_directory_count": len([
            p for p in Path(args.asset_root).expanduser().glob("*") if p.is_dir()
        ]),
    }
    for name, relpath in HANDS.items():
        path = reference / relpath
        result["hands"][name] = urdf_info(path) if path.exists() else {"missing": str(path)}

    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        output = Path(args.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")

    expected = {"linkerhand_o6": 6, "xhand": 12, "wujihand": 20}
    failed = not result["isaac_cuda"].get("isaacgym", False)
    failed |= args.require_cuda and not result["isaac_cuda"].get("cuda", False)
    failed |= any(not ok for ok in result["modules"].values())
    failed |= any(result["hands"].get(k, {}).get("active_joints") != v for k, v in expected.items())
    failed |= result["dataset"]["file_count"] < 100 or bool(result["dataset"]["invalid"])
    failed |= result["asset_object_directory_count"] < 100
    print("RETARGET_PREFLIGHT=" + ("FAIL" if failed else "PASS"))
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
