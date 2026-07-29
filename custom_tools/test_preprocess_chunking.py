"""CPU-only checks for trajectory chunk slicing and output reassembly."""

import numpy as np
from pathlib import Path
from types import SimpleNamespace
import tempfile

import custom_tools.preprocess_graspm3 as preprocessing
import custom_tools.preprocess_graspm3_isolated as isolated


def fake_output(indices, raw_count):
    retained = len(indices)
    return {
        "obs": np.zeros((retained, 70, 3), dtype=np.float32),
        "vis_unscale_actions": np.zeros((retained, 70, 28), dtype=np.float32),
        "success_idx": np.asarray(indices, dtype=np.int64),
        "selection_metric": "official_final",
        "official_final_success_idx": np.asarray(indices, dtype=np.int64),
        "ever_task_success_idx": np.asarray(indices, dtype=np.int64),
        "lift_30cm_idx": np.asarray(indices[:1], dtype=np.int64),
        "maximum_lift": np.zeros(raw_count, dtype=np.float32),
        "obj_rotmat": np.zeros((retained, 3, 3), dtype=np.float32),
        "obj_scale": np.ones(retained, dtype=np.float32),
        "grasp_seqs": np.zeros((retained, 70, 28), dtype=np.float32),
        "hand_pcds": np.zeros((retained, 70, 2, 3), dtype=np.float32),
        "obj_pcds": np.zeros((retained, 70, 2, 3), dtype=np.float32),
    }


def main():
    preprocessing.np = np
    raw = {
        "grasp_seqs": np.zeros((7, 70, 28), dtype=np.float32),
        "obj_rotmat": np.zeros((7, 3, 3), dtype=np.float32),
        "obj_scale": np.ones(7, dtype=np.float32),
        "obj_code": "core-mug-test",
    }
    chunk = preprocessing.subset_raw_trajectories(raw, 2, 5)
    assert chunk["grasp_seqs"].shape[0] == 3
    assert chunk["obj_rotmat"].shape[0] == 3
    assert chunk["obj_scale"].shape[0] == 3
    assert chunk["obj_code"] == raw["obj_code"]

    first = fake_output([0, 2], 4)
    second = fake_output([5, 6], 3)
    merged = preprocessing.merge_processed_chunks([first, second], 7)
    assert merged["grasp_seqs"].shape[0] == 4
    assert merged["maximum_lift"].shape[0] == 7
    assert merged["official_final_success_idx"].tolist() == [0, 2, 5, 6]
    assert merged["success_idx"].tolist() == [0, 2, 5, 6]

    original_run = isolated.subprocess.run
    calls = []

    def fake_run(command, **_kwargs):
        start = int(command[command.index("--trajectory-start") + 1])
        end = int(command[command.index("--trajectory-end") + 1])
        output = Path(command[command.index("--output-file") + 1])
        calls.append((start, end))
        if end - start > 4:
            return SimpleNamespace(returncode=1)
        output.write_bytes(b"complete")
        return SimpleNamespace(returncode=0)

    try:
        isolated.subprocess.run = fake_run
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                selection="official_final", min_free_vram_mb=1, seed=0)
            paths = isolated.run_chunk_with_fallback(
                args, Path("worker.py"), {}, Path("input"), Path(directory),
                "core-camera-test", 0, 8)
            assert [path.name for path in paths] == [
                "chunk_0000_0004.npy", "chunk_0004_0008.npy"]
            assert calls == [(0, 8), (0, 4), (4, 8)]
    finally:
        isolated.subprocess.run = original_run
    print("PREPROCESS_CHUNKING_TEST=PASS")


if __name__ == "__main__":
    main()
