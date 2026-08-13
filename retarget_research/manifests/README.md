# 正式实验清单目录

这里的CSV、抽样manifest、哈希lock和审计JSON均由本机完整GraspM3数据生成，包含绝对数据路径，因此不提交Git。

复现顺序见`../FORMAL_1000_RUNBOOK.md`：先运行`build_embedded_category_map.py`和`build_inventory.py`，再用固定seed运行`build_manifest.py`，最后用`freeze_formal_experiment.py`生成lock。

正式结果必须与本机生成的lock共同验收；不要手工编辑生成文件，也不要用参考仓库的41物体示例代替正式100物体清单。
