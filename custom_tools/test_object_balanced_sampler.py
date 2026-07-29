"""CPU test for equal expected sampling mass per object."""

from collections import defaultdict

import numpy as np

from custom_tools.train_bc import build_object_balanced_sampler


class DummyDataset:
    def __init__(self):
        self.data = {
            'obj_code_idx': np.asarray([1, 1, 2, 3, 3, 3], dtype=np.int64),
        }
        self.is_flat = True
        self.num_frame = 2
        self.obj_code_name_list = ['unused', 'object_a', 'object_b', 'object_c']

    def __len__(self):
        return len(self.data['obj_code_idx']) * self.num_frame


def main():
    dataset = DummyDataset()
    sampler = build_object_balanced_sampler(dataset, seed=2025)
    sample_objects = np.repeat(dataset.data['obj_code_idx'], dataset.num_frame)
    mass = defaultdict(float)
    for object_index, weight in zip(sample_objects, sampler.weights.tolist()):
        mass[int(object_index)] += weight
    values = list(mass.values())
    assert len(values) == 3
    assert max(values) - min(values) < 1e-12, mass
    assert len(sampler) == len(dataset)
    print('OBJECT_BALANCED_SAMPLER_TEST=PASS mass={}'.format(dict(mass)))


if __name__ == '__main__':
    main()
