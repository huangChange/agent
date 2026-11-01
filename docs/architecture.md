# gsforge 架构设计

3DGS 场景 → 可仿真的 mesh 资产（GLB / PLY 为主，URDF / MJCF 可选），带自动修复 agent。

## 0. 已确认的约束

| 项 | 结论 | 对设计的影响 |
|---|---|---|
| 高斯格式 | 标准 3DGS，**可以拿到原始图像和相机位姿** | 纹理主路径是“原图投影烘焙”（`bake_from_capture`），高斯颜色只做兜底；有相机时，up 方向优先从相机推出 |
| 场景 mesh | 算法生成，大部分可用，偶尔质量差 | 只作为几何候选之一（`extract_scene_mesh_crop`），和其它候选一起按 QA 分数择优，**不当真值** |
| 交付 | GLB、PLY 必须；部分用于仿真；关节物体（抽屉）最好有 | GLB 按节点层级导出（每个 link 一个节点），PLY 带顶点色；`sim` profile 再加 URDF / MJCF + 凸分解碰撞体；数据模型已支持关节 |
| 部署 | 云主机 + LLM API | 无状态 worker + 对象存储（S3 兼容）+ Ray；LLM 调用全局限流，按实例记成本 |

## 1. 总体架构

```
            ┌──────────────── 确定性主干（处理绝大多数物体）────────────────┐
JobSpec ──▶ prepare ─▶ split ─▶ geometry ─▶ align ─▶ texture ─▶ material ─▶ physics ─▶ export
             │          │          │          │         │                       │
             └──────────┴──────────┴─ 每个阶段后：QA gates（配置驱动的阈值）───┘
                                              │ 硬指标失败
                                              ▼
                          Repair agent（Claude，observe → think → act → verify）
                          · 看：多视角渲染对比图 + 指标报告
                          · 做：调用注册工具 / 在沙箱里写 Python
                          · 交：submit → 流水线**独立**复评，通过才采纳
                                              │ 超预算 / 无法修复
                                              ▼
                                   needs_review → 人工审核队列
```

几条核心原则：

1. **确定性优先，agent 兜底。** 成本和吞吐取决于确定性主干的通过率；agent 只处理失败的物体。
2. **执行者不给自己打分。** agent 只能提交候选；评估由流水线的 evaluator 完成。VLM critic 是独立请求，看不到 agent 的推理过程。
3. **一切产物不可变、内容寻址。** 工具之间只传 `ArtifactRef`；缓存、断点续跑、跨机器并发都由此而来。
4. **生成的内容要标注。** 所有补全工具都报告 `generated_ratio`，写进 manifest，并影响分级（A 级要求 ≤ 15%）。

## 2. 目录结构

```
configs/                    默认配置、交付 profile（sim / production）、先验表（尺寸、材质）
src/gsforge/
  core/                     领域模型（pydantic）、配置、错误类型、哈希
  io/                       3DGS ply、相机（transforms.json）、mesh 读写
  storage/                  内容寻址存储：本地 / S3 兼容；命名对象（checkpoint、cache）
  tools/                    确定性工具：统一契约 + 注册表（流水线和 agent 共用）
    segmentation/           清洗、实例拆分、支撑面补洞、接触底面闭合
    geometry/               提取候选（density MC / 场景 mesh 裁剪 / GPU TSDF）、修复、对齐
    texture/                顶点色迁移、UV 展开 + 烘焙、原图投影烘焙（GPU）
    material/ physics/      PBR + 物理参数、碰撞体 + 质量惯量
    inspect.py              agent 的“眼睛”：多视角对比渲染、mesh 报告
  render/                   CPU 光栅化 / splat（任意 worker 可用），gsplat GPU 渲染
  qa/                       指标、配置化 gates、阶段 evaluator、分级
  agent/                    LLM 客户端、修复循环、沙箱、critic、角色（roles.py + prompts/）
    skills/                 skill 定义（每个一个目录）、loader、候选脚本库（harvest.py）
  pipeline/                 阶段实现、编排器（checkpoint、并发、交付）、装配、job 导入
  runtime/                  执行后端（本地进程池 / Ray）、带缓存的 ToolRunner
  exporters/                GLB / PLY / OBJ、URDF / MJCF
  evals/                    合成真值场景 + benchmark
tests/                      单元 + agent 循环（脚本化 LLM）+ 端到端
docker/sandbox.Dockerfile   生产沙箱镜像
```

## 3. 工具契约

```python
class SplitParams(BaseModel):          # 参数模型 = 校验 + 自动生成 agent 的 tool schema
    gaussians: ArtifactRef             # agent 侧显示为 "art_xxxxxxxxxxxx"
    up: tuple[float, float, float] = (0, 0, 1)

@tool("split_instances", stages=("split",), version="1.5.0")
def split_instances(ctx: ToolContext, p: SplitParams) -> ToolResult:
    """docstring 就是 agent 看到的工具描述"""
```

- 纯函数，不修改输入；`version` 变化会让缓存失效，并写入 lineage。
- `resource` 标注 cpu / gpu / io，由后端调度到对应资源池。
- 同一个注册表服务流水线和 agent。agent 临时写的代码如果反复成功，就**人工 review 后改写成工具**（见 §5.4），之后不再需要临时生成代码。

## 4. 各阶段

| 阶段 | 做什么 | 关键点 |
|---|---|---|
| prepare | `estimate_up`、`clean_gaussians` | up：主平面法线定轴，相机朝向（或地面所在的一端）定符号；RANSAC 带宽取自采样密度，而不是场景尺寸 |
| split | `split_instances` → `patch_support_surface` → `close_contact_base` | 用“最高的强水平层”作支撑面；用高斯朝向区分“桌面”和“遥控器侧壁底部”；补桌面被遮挡的区域；闭合物体从未被观测到的底面 |
| geometry | 每个候选都跑：提取 → repair → smooth → 减面 → repair，按 QA 分数择优 | density MC：分辨率不超过采样密度，模糊核按平均间距设置，再用距离场补偿偏移，保证始终水密 |
| align | `canonicalize_pose` | 米、+Z 向上、原点在底面中心；世界位姿写入 scene.json，方便重建场景 |
| texture | 顶点色 → 有 GPU 和原图时做原图投影烘焙，否则把顶点色烘进 UV | 未被观测的 texel 回退到顶点色，并计入 generated_ratio |
| material | `assign_material` | 材质类别来自 critic（VLM）或类别先验；PBR 和物理参数查表 `configs/priors/materials.yaml` |
| physics | `build_collision` → `sim_drop_test` | 凸包或 CoACD（在减面后的代理网格上做）；质量、质心、惯量由**未展 UV 的水密几何**的体积 × 密度算出；MuJoCo 跌落测试检查能否静止、是否倾倒、是否下沉 |
| export | GLB（+Y up，PBR，按 link 分节点）、PLY、实例高斯 PLY、URDF / MJCF、manifest | manifest 里有分级、QA 报告、生成比例和 lineage |

## 5. Agent

### 5.1 循环
`agent/loop.py`：手写循环（不用 SDK 的 tool runner），因为每一轮之间要插入独立评估、预算检查、轨迹落盘，工具还要派发到远端 worker。

- 观察：失败的 gate 报告 + mesh / 高斯同机位对比图 + 已知事实 + 同类失败的历史成功脚本。
- 工具：本阶段注册工具 + `run_python` + `evaluate` + `view` + `submit` + `escalate`。
- 结束：`submit` 通过独立复评，或 `escalate`，或预算耗尽（轮数 / 工具调用数 / 美元 / 时长）。
- 历史消息只追加、不改写（thinking block 回放的要求，也能提高 prompt cache 命中率）。

### 5.2 模型调用（`agent/llm.py`）
- `claude-opus-5-5`，adaptive thinking，修复用 effort=high，critic 用 medium。
- 流式 + `get_final_message()`；client tools 开启 `eager_input_streaming`，所有输入在执行前过 pydantic 校验。
- 顶层自动 prompt caching：工具列表和 system prompt 按字节稳定（工具按名称排序）。
- Claude API 上开启服务端 fallback（`fallbacks: "default"`）；Bedrock / Vertex 上用配置关闭。
- 进程内信号量限制并发；SDK 自带对 429 / 5xx 的重试。

### 5.3 沙箱（`agent/sandbox.py`）
- 生产：docker，`--network none`、只读根文件系统、非 root、CPU / 内存 / pids 限制，容器里没有 API 凭证。
- 开发：子进程（`python -I`、清理过的环境变量、超时）。**不是安全边界**，静态检查只能拦住明显的误用。

### 5.4 Skills 与技能沉淀（`agent/skills/`）
- **Skill**（`agent/skills/<name>/SKILL.md` + 参考文件）：审核过的、需要模型判断的处理流程。system prompt 里只放当前阶段的目录（名字 + 一句话描述），全文由 agent 用 `load_skill` 按需拉取。用到的 `name@version` 会记进 trace、StageRecord 和 manifest 的 lineage。定义在 worker 启动时校验；`agent.skill_dirs` 可以挂载额外目录。编写规范见 `agent/skills/README.md`。
- **候选脚本**（`agent/skills/harvest.py`，数据存在产物存储里）：被采纳的、用过代码生成的修复，会按“问题签名”（阶段 + 失败的检查项）存下来。之后遇到同一签名，把历史脚本作为参考注入。
- 升级路径：定期 review 高频签名。能确定性执行的写成 tool，还需要判断的写成 skill。**候选脚本不会自动进入流水线，也不会自动变成 skill。**

### 5.5 Critic（`agent/critic.py`）
独立的 VLM 评审，结构化输出（JSON schema）：类别、材质、忠实度、完整度、问题列表。结果作为 **soft** 检查（VLM 判断有噪声），同时补全未知的 label 和材质。

## 6. 质量体系

- 几何：双向 F-score（recall = 观测到的有没有被覆盖；precision = 有没有凭空多出来），容差 = max(1% 包围盒对角线, 1.5 × 高斯间距)，再加水密、非流形边、连通分量数。
- 对齐：底面在 z=0，尺寸合理性（类别先验）。
- 纹理：有没有贴图、生成比例。
- 物理：质量、碰撞体数量。
- 分级：A（全部通过，生成比例 ≤ 15%）/ B（有 soft 失败）/ C（需人工）/ reject。

**评测**：`gsforge eval` 生成带真值的合成场景（桌 + 遥控器 + 杯子，按真实情况模拟遮挡），跑完整流水线后和真值比对。当前结果（无 agent）：

| 实例 | 方法 | 与真值 F-score | 分级 | 说明 |
|---|---|---|---|---|
| table | density_mc | 0.994 | A | 遥控器和杯子下方的桌面已补齐；17.5 kg（真值约 18 kg）；跌落测试通过 |
| mug | density_mc | 0.997 | B | 底面（16%）为生成；跌落测试通过 |
| remote | density_mc | 0.994 | B | 底面（34%）为生成；跌落测试通过 |

（没有 critic 时 mug 和 remote 的材质会落到默认的 plastic；接入 critic 后由 VLM 判断。）

prompt、模型或工具版本有任何改动，都要先跑这个 benchmark 做回归。下一步是把真实采集的样本经人工审核后加进来，作为 gold set。

## 7. 性能与部署

- **并发粒度**：场景 → 实例，实例之间完全独立（`runtime.max_concurrent_instances`）。
- **资源池**：CPU 工具进程池；GPU 工具每张卡一个单槽进程池（绑定 `CUDA_VISIBLE_DEVICES`）；LLM 全局限流。生产用 Ray（`RayBackend`），每次工具调用就是一个带 num_cpus / num_gpus 的 task。
- **缓存**：key = (工具名, 版本, 参数, 输入 sha256)，只改下游参数时上游直接命中。
- **断点续跑**：每个阶段结束后写 checkpoint；同一个 job 重跑会从断点继续。
- **存储**：S3 兼容对象存储 + 本地只读缓存；写入幂等（按 hash）。
- **成本**：每个实例每个阶段都有美元上限；成本记在 StageRecord 和 scene.json 里。

## 8. 实现状态（诚实清单）

| 模块 | 状态 |
|---|---|
| 核心模型、存储、配置、工具注册、缓存、checkpoint、编排、导出、QA、合成评测 | ✅ 已实现，测试覆盖 |
| 实例拆分、支撑面补洞、底面闭合、density MC、场景 mesh 裁剪、repair、对齐、顶点色、UV 烘焙、碰撞体（凸包 / CoACD）、MuJoCo 跌落测试 | ✅ 已实现，合成场景验证；生成的 MJCF 已用 MuJoCo 加载并仿真 |
| Agent 循环、沙箱、技能库、critic | ✅ 已实现；循环用脚本化 LLM 测过，**还没有对真实 Claude API 跑过** |
| GPU TSDF（gsplat）、原图投影烘焙（nvdiffrast）| ⚠️ 代码已写，本机没有 GPU 依赖，**未运行验证** |
| S3 存储、Ray 后端、docker 沙箱 | ⚠️ 代码已写，未在云上验证 |
| 关节：数据模型 + URDF / MJCF 导出 | ✅ 已实现 |
| 关节：自动检测（抽屉 / 门的部件分割、轴和行程估计）| ❌ 待做（见路线图） |
| 去光照（albedo 分离）、生成式形状补全、文字 / logo 修复 | ❌ 待做（见路线图） |

## 9. 路线图

1. **M1 打通真实数据**：在 GPU worker 上验证 `extract_tsdf_render` 和 `bake_from_capture`；用 10–20 个真实场景标定 QA 阈值；做人工审核 UI。
2. **M2 agent 上线**：真实 API 联调，统计各阶段确定性通过率、agent 修复率、单实例成本；把审核反馈加入评测集。
3. **M3 外观**：intrinsic decomposition 去光照；多视角一致的扩散补全（先在渲染视角补，再反投影，不直接在 UV 上补）；文字区域检测 → 超分 → VLM 识别 → 重绘，生成区域写入置信度 mask。
4. **M4 关节**：VLM 提出可动部件假设 → 多视角 2D 分割提升到 3D 部件 → 估计关节类型、轴和行程（抽屉 = 沿前向平移，门 = 绕铰链边旋转）→ 补全被遮挡的内部（抽屉箱体）→ 在 MuJoCo 里做运动范围检查。
5. **M5 规模化**：Ray autoscaling、队列优先级、USD 导出（Isaac Sim）、按客户配置交付 profile。
6. **合规**：品牌 logo / 商标的处理策略需要法务确认（建议默认模糊或去除可识别商标，按客户需求开启）。
