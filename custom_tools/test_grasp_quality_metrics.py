import torch
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from custom_tools.grasp_quality_metrics import (
    instantaneous_official_success, stable_official_success)


def main():
    goal = torch.tensor([[0.0, 0.0, 0.9]]).repeat(4, 1)
    objects = torch.tensor([
        [0.0, 0.0, 0.781],
        [0.0, 0.0, 0.779],
        [0.0, 0.0, 0.900],
        [2.0, 0.0, 0.900],
    ])
    assert instantaneous_official_success(objects, goal).tolist() == [
        True, False, True, False]
    terminal = torch.tensor([
        [True, True, True],
        [True, False, True],
        [True, True, True],
    ])
    stable = stable_official_success(
        terminal,
        maximum_height=torch.tensor([0.30, 0.30, 0.30]),
        final_height=torch.tensor([0.29, 0.29, 0.25]))
    assert stable.tolist() == [True, False, False]
    print("GRASP_QUALITY_METRICS_TEST=PASS")


if __name__ == "__main__":
    main()
