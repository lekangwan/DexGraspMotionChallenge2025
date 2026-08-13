# 环境与复现状态

## 独立环境

```bash
conda activate hand-retarget
```

该环境从旧 `dexgrasp` 环境克隆，随后只在新环境中将NumPy调整为1.23.5，并安装NLopt 2.7.1与autograd 1.6.2。旧环境没有被升级或改写。参考仓库要求Python 3.8、Isaac Gym Preview 4、NumPy<1.24；当前新环境Python为3.8，PyTorch为2.4.1+cu121，与参考README的2.0.1+cu117不完全相同，所以GPU仿真恢复后必须重新做兼容性验证。

## 已通过的CPU冒烟测试

从参考数据中只取一个物体的一条轨迹，XHand SLSQP运行70帧并成功保存 `(1,70,18)` 输出。正式测试将 `iter_num` 从2恢复到100，并使用冻结manifest。

## 当前未通过项

`nvidia-smi`无法连接驱动，`torch.cuda.is_available()`为False，因此Isaac Gym物理重放尚未验证。系统存在CUDA 12.1 Conda工具链，但shell没有系统级 `nvcc`；TorchSDF和PyTorch3D在克隆环境中已有预编译包，若后续出现符号错误，不应直接覆盖安装，而应先记录torch/CUDA/扩展版本再统一重建。

## 正式预检

```bash
conda activate hand-retarget
python retarget_research/scripts/preflight.py \
  --reference-root retarget_research/reference/HandRetargetTask2026 \
  --dataset-root /path/to/full/seq \
  --asset-root /path/to/full/object_assets \
  --output retarget_research/logs/preflight.json
```

GPU恢复后还要单独执行：

```bash
python -c "import isaacgym; from isaacgym import gymapi; import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
python /home/lekangwan/isaacgym/python/examples/joint_monkey.py
```

