# gsforge

把分割后的 3DGS 场景转换成可仿真的 mesh 资产（GLB / PLY，可选 URDF / MJCF），并在质量不达标时由 Claude agent 自动修复。

设计文档：[docs/architecture.md](docs/architecture.md)

## 快速开始

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe -e ".[dev,geo]"     # Linux/macOS: .venv/bin/python

gsforge tools                          # 列出注册的工具
gsforge eval --out .gsforge/eval       # 合成真值 benchmark（不调用 LLM）
pytest                                 # 单元 + agent 循环 + 端到端测试
```

处理一个场景（job 格式见 [examples/job.example.yaml](examples/job.example.yaml)）：

```bash
gsforge run path/to/job.yaml --no-agent        # 只跑确定性流水线
gsforge run path/to/job.yaml                   # 失败的阶段交给 repair agent（需要 Claude API 凭证）
gsforge run path/to/job.yaml --profile sim     # 额外导出 URDF / MJCF + 凸分解碰撞体
```

交付目录（`export.out_dir/<scene_id>/`）：

```
scene.json                    实例列表、在原场景中的位姿、分级、各阶段状态、成本
<instance_id>/
  asset.glb                   +Y up，PBR，米制
  asset.ply                   +Z up，顶点色
  gaussians.ply               该实例的 3DGS（规范坐标系下）
  sim/asset.urdf, asset.xml   （sim profile）
  manifest.json               分级、QA 报告、生成比例、lineage（工具 / 模型 / 配置版本、输入哈希）
```

## 配置

`configs/default.yaml` ← `configs/profiles/<profile>.yaml` ← 环境变量 `GSFORGE__SECTION__KEY=value`。

QA 阈值、阶段失败策略（agent / review / fail）、agent 预算、模型和 effort 都在配置里，修改不需要改代码。

## 可选依赖

| extra | 用途 |
|---|---|
| `geo` | xatlas（UV）、pymeshlab（大洞修补）、manifold3d、coacd（凸分解） |
| `gpu` | torch + gsplat（TSDF 候选）；原图投影烘焙还需要 nvdiffrast |
| `cloud` | ray、boto3（S3 兼容存储） |
| `sim` | mujoco（仿真验证） |
