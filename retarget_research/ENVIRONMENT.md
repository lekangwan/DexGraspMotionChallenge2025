# 环境与复现状态

## 独立环境

```bash
conda activate hand-retarget
```

该环境从旧 `dexgrasp` 环境克隆，随后只在新环境中将NumPy调整为1.23.5，并安装NLopt 2.7.1与autograd 1.6.2。旧环境没有被升级或改写。参考仓库要求Python 3.8、Isaac Gym Preview 4、NumPy<1.24；当前新环境Python为3.8，PyTorch为2.4.1+cu121，与参考README的2.0.1+cu117不完全相同，所以GPU仿真恢复后必须重新做兼容性验证。

## 已通过的CPU冒烟测试

从参考数据中只取一个物体的一条轨迹，XHand SLSQP运行70帧并成功保存 `(1,70,18)` 输出。正式测试将 `iter_num` 从2恢复到100，并使用冻结manifest。

## 最终验证状态

用户终端中的NVIDIA RTX 4060 Laptop GPU可用，PyTorch网络训练使用CUDA；Residual PPO三手各300轮训练已经完成。Isaac Gym接触仿真统一使用CPU PhysX，以避免把网络训练设备和物理设备混为一谈；GPU图形设备只用于报告视频的Isaac相机渲染。

正式机器CPU为Intel Core i7-14650HX。最终软件与checkpoint哈希见`reports/FINAL_EXPERIMENT_METADATA.json`。Codex沙箱有时无法访问NVIDIA驱动，这只影响沙箱内探测，不代表用户终端或正式实验没有使用GPU。

## 正式预检

```bash
conda activate hand-retarget
python retarget_research/scripts/preflight.py \
  --reference-root retarget_research/reference/HandRetargetTask2026 \
  --dataset-root /path/to/full/seq \
  --asset-root /path/to/full/object_assets \
  --output retarget_research/logs/preflight.json
```

需要重新部署时可执行：

```bash
python -c "import isaacgym; from isaacgym import gymapi; import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
python /home/lekangwan/isaacgym/python/examples/joint_monkey.py
```
