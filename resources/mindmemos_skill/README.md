# MindMemOS Skill benchmark resources

此目录只保存适合 Git 版本管理的小型、可复现实验资产；完整数据集下载到被 `.gitignore` 排除的
`data/mindmemos_skill/`。

```text
resources/mindmemos_skill/
├── datasets/
│   ├── alfworld/{split_manifest.json,splits/{train,val,test}/items.json}
│   ├── livemath/{split_manifest.json,splits/{train,val,test}/items.json}
│   └── spreadsheetbench/{split_manifest.json,splits/{train,val,test}/items.json}
└── skills/
    ├── alfworld/SKILL.md
    ├── livemath/SKILL.md
    └── spreadsheetbench/SKILL.md
```

这些划分和初始 Skill 与 `skillOpt` 仓库中的论文实验资产对齐。各 `split_manifest.json` 记录上游仓库、
固定 revision、源文件、划分方式和数量；`splits/` 使用 `skillOpt` Git 版本中的轻量 manifest，而不是其
本地 materialized 数据文件，只包含任务 ID/路径及少量索引元数据，不包含大型 benchmark payload。

数据来源与许可证以各上游项目为准：

- ALFWorld：<https://github.com/alfworld/alfworld>，通过官方 `alfworld-download` 下载。
- LiveMathematicianBench：<https://huggingface.co/datasets/LiveMathematicianBench/LiveMathematicianBench>。
- SpreadsheetBench Verified-400：<https://huggingface.co/datasets/KAKA22/SpreadsheetBench>。

下载命令见 `docs/skill_algo_develop/experiment_runner.md`。不要把 `data/mindmemos_skill/`、Hugging Face 缓存或
运行输出复制进本目录。
