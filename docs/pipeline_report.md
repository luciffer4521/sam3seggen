# 拆件管线技术报告

几何决定每一条边界；语言只负责取名。CLI、Python、HTTP 共用 `pipeline.PipelineOptions`，默认一条龙：

```
paint → guidance → split → units → merge → complete → bake
```

实现上 `sample_0`（`full_seg`）必须先跑：它是求交的参考网格，也是平涂的色源。引导图插在剩余 6 次采样之前，坏提示词不用先付完整 GPU 账。`merge=off` 跳过命名仍可修复；`complete=off` 停在开口的 `parts.glb`。当前默认是 `complete=hybrid`。

契约文件：`pipeline.py`。编排：`segment_parts.py`。命名与修复：`merge_parts.py`。HTTP：`serve_api.py`。

---

## 1. 管线路径

三个入口读同一份开关，字段名对齐 dataclass：


| 入口           | 文件                            | 角色                    |
| ------------ | ----------------------------- | --------------------- |
| CLI / Python | `segment_parts.segment_parts` | 全程编排                  |
| 只重命名         | `merge_parts.merge_parts`     | 对已有 `work/` 再投票、导出、修复 |
| HTTP         | `POST /segment`               | multipart 上传，串行 GPU 锁 |


实际执行顺序（`segment_parts.py`）：

```
full_seg sample_0          → work/sample_00/seg.glb
    ↓
[有提示词] render → 可选平涂 → SAM3 → guidance overlays
    ↓
full_seg sample_1 … N-1    → 默认 N=7，方位 [0, 30, -30, 15, -15, 0, 30]
    ↓
meet_samples               → atoms.npy / atoms.glb
    ↓
split_units + fuse 双壳    → units.npy
    ↓
merge != off               → IoU 投票命名 → parts.glb → complete → bake
merge == off               → 匿名 unit_XX 导出，仍可 complete
```

作业目录（HTTP：`$SEGVIGEN_JOBS_DIR/{job_id}/`）：

```
input.glb
parts.glb
parts.json
work/
  sample_00..06/{seg.glb, render.png, input.vxz, azimuth.json}
  views/  views_flat/  guidance/*.png
  masks_<sha1[:10]>.npz
  atoms.npy  atoms.glb  atoms_report.json
  units.npy
  labels.npy  label_names.json  vote_report.json
complete/
  boxes.json  boxes.glb  open_instances.glb
  xpart_instances.glb  xpart_parts_raw.glb  xpart_parts.glb
  holopart_instances.glb     # 仅当有实例被换成 HoloPart
  hybrid_instances.glb  decisions.json
```

---



## 2. 各阶段技术路线



### 2.1 paint — 给灰模临时上色

无色模型上 SAM3 经常 `no instance`，背面整片掉进 `unassigned_to`。`flat_paint=auto` 时，若源渲染饱和度 `< 8`，就用 **sample_0 的 full_seg 部件色** 铺一层临时平色再给 SAM3。只服务命名，不改几何。

#### 部件色从哪来

颜色是 SegviGen Lora自己画在 `work/sample_00/seg.glb` 上的：

```
源 glb → vxz 形状编码
    ↓
按 sample_0 方位渲染一张条件图（DINOv3；transforms.json + azimuth）
    ↓
full_seg.ckpt 在 TRELLIS.2 的 texture flow 上采样 tex_slat
    ↓
tex_decoder → 体素属性，其中 base_color[:3] 就是预测的部件色
    ↓
slat_to_glb remesh，把属性烘进 4096 UV，写成 baseColorTexture
    ↓
导出 work/sample_00/seg.glb
```

`full_seg` 训练目标就是「每个几何部件一块近似常值色」。主线 **不传** `--two_d_map`，所以这块色只来自：形状 + 这一次条件视角的外观。注释写明：条件相机落在哪一面，部件色就从哪一面推（`inference_full.py`）。因此 sample_0 必须先跑——后面的平涂和求交都以它的网格和这张纹理为参考。

直接拿 `seg.glb` 上的颜色给 SAM3 **不够用**。机器人上每个部件都是略有差别的绿，边界在渲染里几乎看不见。`flat_paint.part_colors` 的处理是：

1. `face_base_colors`：在 UV 面心采样 `baseColorTexture`，得到每面 RGB。
2. `cluster_parts`（与求交同一套）：颜色 `//8` 量化后按面积从大到小贪心聚类，`color_tol=20`（不用 parts_rebake 的 40：40 会把臂并进躯干，臂掩码从 9834 掉到 3050 px）。
3. 聚类 id 循环贴到 `PAINT_COLORS`：12 个高饱和、色相差得开的色（红 / 蓝 / 绿 / 黄…）。件的个数仍是 full_seg 预测的，只是互相能分开。
4. `SEG_TO_CAMERA = diag(1,-1,-1)` 后栅格化到投票相机，**不走 Blender**。再渲染会把 `seg.glb` 里的轴交换带回来；栅格化与源渲染剪影 IoU = 1.000。

必须 **flat**：同一套色乘上渲染亮度后，机器人背面 `torso` 从 3721 px 掉到 573 / `no instance`。

- 饱和度：`mean(max−min)`，只计 `alpha>16`。门槛 `COLORLESS_SATURATION = 8.0`。
- 实测：无贴图机器人 0.25–0.42（平涂）；有 albedo 的角色 43–53（跳过）。
- 默认 `flat_paint=auto`；`on` 强制；`off` 永不。产物：`work/views_flat/`。



### 2.2 guidance — 多视角 SAM3 掩码

固定四角 3/4 渲染源模型 → 可选平涂 → SAM3 每概念一张掩码 → 审阅叠加图。

- 相机默认 `45,135,225,315 × 仰角 10°`，半径 2，分辨率 512。正对 90° 漏胸口；仰 35° 躯干挡腿脚。
- 概念库 v3 `bank.pt`，阈值 **0.4**；无库 **0.3**。
- `overlay_v3`：面积从小到大、互斥。大 `arm` 盖不住背包。
- 某概念所有视角都没有：默认 **skip**（`strict_parts=true` 才整单失败）。
- 产物：`work/views/`、`masks_*.npz`、`work/guidance/*.png`。

语言不切几何。掩码只给后面的单元投票用。

### 2.3 split — 无提示 full_seg 求交

N 次 **不带 2D 图** 的 `full_seg`（`ckpt/full_seg.ckpt`，不是 `full_seg_w_2d_map`）。只有每一次采样都同色的面才同原子。过分割可合，欠分割不可挽回。

单次流程（`inference_full.py`）：glb → vxz → TRELLIS.2 latent → DINOv3 条件（`transforms.json` + azimuth）→ flow sample → remesh GLB → 丢掉离体碎片。

求交（`meet_samples.py`）：

1. 每样本 `cluster_parts`，`color_tol=20`。
2. 非参考网格：最近面心把标签投到 sample_0。
3. `mirror=auto`：bbox 中面反射，共享 ≥0.9、容差 0.01×对角线。命中则每样本再加一份镜像标签。
4. 跨样本标签元组 `np.unique` → 生原子。
5. `absorb_small_fragments`，下限随粒度（medium=300 面），迭代 5 次。
6. 同标签再按连通分量切开。

默认 `samples=7`，`azimuth_jitter=30`。多样本买的是 **方差**（肩甲不要和双臂焊死），不是更细。`samples=1` 可复现旧的单次染色。

产物：`work/sample_XX/seg.glb`、`atoms.npy`、`atoms.glb`、`atoms_report.json`。

### 2.4 units — 连通块 + 双壳合并

原子内焊死连通分量；双壳内壁并进外壳。必须在命名前做。

- 分量 `< min_unit_faces`（medium=600）并进同原子最近大块。
- 双壳：外向度 ≥0.7 vs ≤0.3；质心距 <0.08×对角线；盒包含 ≥0.9；面积比 [0.4, 1.6]。
- 内壁无共边，单独投票会继承最近可见件（机器人背包内衬曾整背变臂）。

产物：`work/units.npy`。

### 2.5 merge — 投票取名

`unit_vote.vote`，**不是**覆盖率决胜：

1. 单元栅格化到各视角（剪影 IoU 应对齐到 1.0 量级）。
2. 视角可见像素 <50 不投。
3. **认领**：`recall = inter/unit ≥ 0.5`。
4. **决胜**：已认领里 **IoU 最高**（最贴合，不是最大）。coverage 会让 `leg` 吃掉 `foot`。
5. **逐视角一票**，多数胜；并列看 IoU 之和。像素池化会被正对相机单独决定。
6. 无票可见单元：只挂在已投票邻件上、且面数 < 0.1×最大邻件 → 并入邻件多数名。否则 → `unassigned_to`。
7. 不可见单元：最近可见面继承。

`unassigned_to` 默认 `"body"`，必须是提示词里的名；不在名单则忽略并丢掉这些面。Swagger 占位 `"string"` 当没填。


| `merge`     | 节点                                              |
| ----------- | ----------------------------------------------- |
| `name`（默认）  | 每提示词一个，同名焊死                                     |
| `unit`      | `{index:02d}_{voted_name}`                      |
| `fragments` | 几何切开留下；面积 < `fragment_share=0.01` 且（无名或与邻同名）才折回 |
| `off`       | `unit_XX`，仍跑 complete                           |


产物：`parts.glb`、`parts.json`、`vote_report.json`。

### 2.6 complete — 封闭实体

`complete_parts` 子进程调度 `xpart_complete.py`（X-Part 独立 venv）。

预处理：按名拆焊死连通分量（两只手各一实例）；盒包含 ≥0.98 的内壁丢弃；面积 < 0.5% 折进最近大件；parts 框与源框三轴尺度差 >5% 直接退出。

- `condition=surface`（默认）：每实例从归属面采 81920 点 `(xyz, normal, sharp=0)`。盒子仍传，定 token。`box` = 盒内裁剪（躯干更容易胀）。
- `octree_resolution=512`，`seed=42`，`parts_per_batch=4`，`num_chunks=50000`。
- 超框 >50% 重抽 `redraws=2`，只留更贴盒的那一抽。

**hybrid**（`hybrid_complete.py`）按 **实例** 换后端，不是整模换：


| 条件   | 阈值                           |
| ---- | ---------------------------- |
| 超框   | 伸出盒外 / 盒宽 的 max > **50%**    |
| 且为大件 | 面积 ≥ **8%** 或某轴 ≥ 源模 **55%** |


同时满足才换成该实例的 HoloPart。机器人实测通常只换躯干：腿大但不超框；手即使飘了也太小。HoloPart 仅在至少一件达标时加载。

`off` 不修；`boxes` 只写提示；`full` 纯 X-Part。

### 2.7 bake — 源 albedo 回烘

开口件：`parts_rebake.py --labels`。封闭件：先把结果改名为 `xpart_parts_raw.glb`，再 `adapt_cage=True` 回烘。

分辨率（底边默认 2048，与 hybrid「大件」面积门槛对齐）：


| 面积占比  | 边长       |
| ----- | -------- |
| < 8%  | 2048     |
| ≥ 8%  | 4096     |
| ≥ 40% | 8192（封顶） |


自适应笼子（仅封闭件）：上限 0.05 / 0.15；贴得近用 0.02 / 0.05。`gap` = 目标顶点到源表面的中位距离。`extrusion = clamp(gap×1.5, 0.02, 0.05)`，`ray = clamp(gap×4, 0.05, 0.15)`。Cycles selected-to-active，16 samples，margin 2。贴图 **PNG pack 进 GLB**。

---



## 3. 当前默认（`pipeline.py`）


| 开关                       | 值              | 作用                     |
| ------------------------ | -------------- | ---------------------- |
| `prompts` 空              | 主体、底座          | 不再默认走匿名 `merge=off`    |
| `merge`                  | `name`         | 同名焊成一节点                |
| `unassigned_to`          | `body`         | 不在提示词里则忽略              |
| `complete`               | `hybrid`       | X-Part，大件超框才换 HoloPart |
| `condition`              | `surface`      | 用拆分归属面约束生成             |
| `flat_paint`             | `auto`         | 灰模才平涂                  |
| `granularity`            | `medium`       | atom 300 / unit 600    |
| `samples`                | 7              | 求交次数                   |
| `azimuth_jitter`         | 30°            | 条件相机抖动                 |
| `color_tol`              | 20             | 染色聚类                   |
| `mirror`                 | `auto`         | 对称再投一份标签               |
| `view_azimuths`          | 45,135,225,315 | 四角 3/4                 |
| `view_elevations`        | 10             | 略仰，不挡腿                 |
| `sam3_threshold`         | 0.4            | 概念库门槛                  |
| `min_recall`             | 0.5            | 投票认领                   |
| `fragment_share`         | 0.01           | fragments 折碎屑          |
| `min_area_share`         | 0.005          | 修复前折小分量                |
| `redraws`                | 2              | 超框重抽                   |
| `texture_size`           | 2048           | 小件底边，大件升 4K/8K         |
| `with_texture` / `reuse` | true           | 回烘；复用缓存                |


粒度预设：fine 150/300，medium 300/600，coarse 800/1600。

---



## 4. 相对原版的修改

「原版」指本仓库早期主线：SegviGen + SAM3 **2D 引导图** 条件生成（`segment_api` / `POST /segment_legacy`），以及随后的单次 `full_seg` + 覆盖率投票（`segment_vote.py`）。不是 Hunyuan 上游本身。

### 4.1 主线从「语言切几何」改成「几何先切、语言只命名」

原版：单视角渲染 → SAM3 画 2D 色图 → `full_seg_w_2d_map` **被这张图引导生成**。一个错像素就能在 3D 撕边。切分模式：

- **stain**：沿染色切，只吸收 <100 面碎块。
- **weld**：切口不变，整块按可见占比换名。
- **refine**：逐像素回写 + 小岛重分。

现版：`full_seg` **不传** `--two_d_map`。提示词只进入 SAM3 投票。`POST /segment` 不再走 2D 路线；旧入口仍在 `POST /segment_legacy`。

### 4.2 单次染色 → 7 次求交

`segment_vote` 相信一次 `full_seg`。单次同视角下，机器人肩甲+双臂可以是一个 24k 面原子，事后切不开。

现版 7 次求交：只有每次都同色才同原子。扫参：5 次不够稳；7 与 9 落在同一套命名部件，9 只多原子。原子下限 150→300：机器人 66→59 原子，命名部件数不变。

### 4.3 覆盖率投票 → IoU + 逐视角计票

原版 `part_vote.assign_parts`：覆盖率门槛 0.25，谁盖得多谁赢 → 脚变腿、臂变躯干。

现版：recall ≥0.5 认领，IoU 决胜，每视角一票。正对相机不能单独吞掉侧面件。

### 4.4 空提示词不再炸出上万匿名件

旧契约：空 prompts **只允许** `merge=off` → 每个几何单元一个节点。高模实测 37631 atoms / **37474 units**（`unit_00`…），Blender 回烘在碎屑上 `Problem saving the bake map`。

现版：空 prompts **一律填 主体、底座**，`merge` / `granularity` 保持请求值。默认 `name` 焊成两块再 hybrid。显式 `merge=off` 仍出匿名单元，这是唯一还会炸出上万件的路径。

### 4.5 新增 `merge=fragments`

几何切开留下；面积 <1% 的无名/同名碎屑折回邻件。同名两大件保持两节点（`name` 会焊在一起）。给椅子横档、独立小件用。

### 4.6 修复从「无 / 纯 X-Part」到 hybrid

早期 complete 只有 `off|boxes|full`。现默认 hybrid：先 X-Part；仅当该实例又大又超框才换 HoloPart。表面条件相对盒条件，机器人躯干跑偏从 65.4% 降到 5.0%。

### 4.7 投票相机

演进：`segment_vote` 八方位×两仰角 → 一度改成 `45,225×10`（躲 90° 漏胸、躲 35° 挡腿）→ 现 **四角 3/4**，两侧都能投到（耳、单侧件）。

### 4.8 回烘质量

封闭件回烘后补上：按面积升到 4K/8K；按中位 gap 收紧笼子；PNG 打进 GLB。避免整模统一 2K + 松笼子把字烘糊、把缝烘脏。

### 4.9 HTTP 可运维

- GPU `threading.Lock()`，第二请求 **409**，不排队。
- AutoDL 网关掐长 `POST` 后浏览器常显示 404：`GET /jobs`、`/jobs/latest`、`/jobs/{id}`、`/health.latest_job` 找回任务。盘上按产物推断阶段。
- Swagger 占位 `string` / `integer` / 空数字当没填；`glb` 当文本 → 422，不再二次 500。
- `unassigned_to` 默认 `body`。



### 4.10 其它相对原版的修复

- 离体碎片清理（`inference_full.drop_offbody_components`）
- 无掩码提示词默认 skip，不再整单失败
- 投票挂接 `hang_share=0.1`（胡子接头，手不接躯干）
- 相机约定标定（渲染器剪影 IoU ≈ 0.987）
- 图集合并 glTF V 轴、metallic / Emission 通道
- `finetune/` LoRA 仍只服务旧 2D-map 域，不在当前主线推理路径上

关键提交（仓库 `git log`）：


| commit    | 变化                                  |
| --------- | ----------------------------------- |
| `c85d5c6` | 起点：SegviGen + SAM3 2D 引导图           |
| `7600395` | 提示词组、vote-merge、自动正面                |
| `8b2ca22` | 过分割再命名成为主线；2D 降为 deprecated         |
| `241be10` | 五段重排；guidance 挪到贵采样前；flat_paint     |
| `4dd26d5` | `condition=surface`；默认 7 次、300/600  |
| `c9e62ee` | fragments、redraws、granularity 名     |
| `8ba467a` | `PipelineOptions` 统一入口；封闭件回烘        |
| `0a739b1` | 无掩码 skip                            |
| `bb2056c` | `complete` 默认 hybrid；prompts 一句逗号分隔 |
| `d15a46a` | 空提示词 → 主体/底座；Swagger 占位忽略           |
| `3031ed6` | 投票相机 → 45,135,225,315 × 10          |


---



## 5. 必读代码


| 文件                                   | 内容                                        |
| ------------------------------------ | ----------------------------------------- |
| `pipeline.py`                        | 契约与全部默认                                   |
| `segment_parts.py`                   | 编排、`sample_azimuths`、`resolve_unprompted` |
| `merge_parts.py`                     | guidance / 投票导出 / `complete_parts`        |
| `inference_full.py`                  | 无图 `full_seg`                             |
| `sam3_multiview.py`                  | 多视角掩码、v3 overlay                          |
| `data_toolkit/meet_samples.py`       | 求交、镜像                                     |
| `data_toolkit/unit_vote.py`          | 单元、双壳、IoU 投票、fragments                    |
| `data_toolkit/parts_rebake.py`       | 烘焙、笼子、贴图尺寸；旧 stain/weld/refine            |
| `flat_paint.py`                      | 饱和度门槛与平涂                                  |
| `xpart_complete.py`                  | 表面条件、超框重抽                                 |
| `hybrid_complete.py`                 | 0.5 / 0.08 / 0.55                         |
| `holopart_complete.py`               | 按需 HoloPart                               |
| `serve_api.py`                       | HTTP、锁、作业恢复                               |
| `prompt_specs.py`                    | 逗号句、`body=head+face`                      |
| `segment_api.py` / `segment_vote.py` | 旧 2D / 单样本 coverage                       |


对外调用说明仍见 [api_split_complete_bake.md](api_split_complete_bake.md)。改默认值改 `pipeline.py` 并重启 `run_serve.sh`，否则 `/health` 仍是旧进程。