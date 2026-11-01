# Agent skills

这里放**审核过的**、agent 按需加载的处理方法。每个 skill 一个目录，目录里是 `SKILL.md` 加可选的参考文件。

## 三种东西各放哪里

| | 什么时候用 | 放在哪 |
|---|---|---|
| tool | 不需要模型做判断就能执行的确定性操作 | `gsforge/tools/`，有测试，有版本号 |
| skill | 需要模型看图、做判断的处理流程，可以引用 tool 和参考脚本 | 这里 |
| 候选脚本 | agent 修复成功时自动记录的代码，还没审核 | 产物存储（`harvest.py`），**不进源码** |

升级路径：先从候选脚本里挑出高频签名，人工 review。能确定性执行的写成 tool，还需要判断的写成 skill。

## 加载机制

- system prompt 里只放当前阶段可用 skill 的 `name: description` 列表；全文由 agent 调用 `load_skill` 获取。
- 用到的 skill 会以 `name@version` 的形式记进 agent trace、`StageRecord.skills_used` 和交付 manifest 的 `lineage.skills_used`。
- 除了内置目录，还可以通过配置 `agent.skill_dirs` 加载其它目录（比如某个客户专用的处理方法）。skill 名字不能重复，重复会直接报错。
- 定义在 worker 启动时就做校验：frontmatter 不合法、名字和目录名不一致、阶段名写错，都会直接启动失败，不会拖到修复进行中才出错。

## 新增一个 skill

1. 复制 `_template/`，改成新的名字（小写字母，用 `-` 连接，必须和 frontmatter 里的 `name` 一致）。
2. `description` 只写一行，要让 agent 从失败的检查项和预览图就能判断是否适用。这一行决定 skill 会不会被用上。
3. 参考脚本必须能在沙箱里运行：输入从 `inputs/manifest.json` 读，输出写到 `outputs/`，不能联网、不能起子进程。
4. 在 `tests/test_skills.py` 里加测试，至少覆盖：能被加载；参考脚本在沙箱里能跑出预期结果。
5. 修改行为时升级 `version`，trace 和 lineage 就能区分新旧版本。
6. 跑一遍 `gsforge eval` 回归测试。

## 写好 skill 的要点

- 先写**什么时候适用**，以及**什么时候不适用**。用错的 skill 比没有 skill 更糟。
- 步骤里写具体的工具名和参数，不要写泛泛的建议。
- 写清楚怎么验证：哪些指标不能变差，预览图里要看什么。
- 凡是有内容是生成的，写清楚要在 summary 里怎么说明。
