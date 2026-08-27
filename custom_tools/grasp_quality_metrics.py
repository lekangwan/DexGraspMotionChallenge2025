"""Success metrics separated from policy and simulator implementation.

Do not import torch here: Isaac Gym must be imported before PyTorch.  These
helpers operate on tensors passed in by the evaluator and need no torch names.
"""


def instantaneous_official_success(object_pos, goal_pos, tolerance_m=0.12):
    goal_distance = (goal_pos - object_pos).norm(p=2, dim=-1)
    in_workspace = (
        (object_pos[:, 0] >= -1.5)
        & (object_pos[:, 0] <= 1.5)
        & (object_pos[:, 1] >= -1.5)
        & (object_pos[:, 1] <= 1.5)
        & (object_pos[:, 2] < 2.0))
    reached = (
        (goal_distance <= float(tolerance_m))
        | (object_pos[:, 2] >= goal_pos[:, 2]))
    return reached & in_workspace


def stable_official_success(
        terminal_official, maximum_height, final_height,
        maximum_drop_m=0.03):
    held = terminal_official.bool().all(dim=0)
    low_drop = maximum_height - final_height <= float(maximum_drop_m)
    return held & low_drop
