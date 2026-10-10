# 灾害情境下的容量感知就医推荐

本项目研究灾害期间患者如何选择医疗设施，以及如何通过容量感知推荐缩短从出发到开始服务的时间。方法将道路通行、压力下的设施偏好和医疗服务排队连接起来：灾害改变各条道路的通行时间，患者根据距离、设施声望与规模形成初始选择，设施的服务能力决定到达后的等待时间。

### 方法概览

1. **构建灾害路网。** 使用 OpenStreetMap 有向路网，根据道路与灾害中心的距离施加通行时间惩罚，计算患者起点到各医疗设施的最短出行时间。
2. **模拟患者选择。** 用个体压力阈值控制出行便利性与设施吸引力的权重，并加入随机偏好。需要核心医院的患者只能选择核心医院；其他患者可以选择医院或诊所。
3. **生成容量感知推荐（Adaptive）。** 按患者出发顺序处理决策，利用此前已承诺的到达时间和设施服务台配置预测等待，比较各候选设施的“出行时间 + 预计等待时间”。仅在符合就医类型、预计收益达到阈值且额外出行未超限时推荐改选，并按设定遵从率更新到达计划。默认收益阈值为 5 分钟，额外出行上限为 10 分钟，遵从率为 80%。
4. **计算实际服务结果。** 所有患者完成选择后，按实际到达时间模拟多服务台先到先服务排队，得到出行、等待、开始服务时间及 60 分钟内开始服务的比例。推荐只使用此前决策的信息，后续患者仍可能更早到达，因此预计收益与实际收益可能不同。
5. **进行配对比较。** 在相同人群、灾害路网和随机偏好下，将 Adaptive 与无干预、医院降权、诊所曝光及最快出行等情景比较，通过多个随机种子评估平均效应与波动。

核心思路是把设施服务能力纳入推荐：更短的出行可能伴随更长的排队，推荐因此同时考虑两者。当前实验使用香港和东京 23 区的 OSM 路网，扫描需求与灾害强度，检验方法的收益及适用条件。设施容量和行为参数目前为示例设定，结果用于机制探索；模型假设推荐平台可观察此前患者的选择，尚需真实服务能力与行为数据校准。

## 快速运行

需要 Python 3.10+。建议在项目虚拟环境中安装：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -v

# 离线合成路网：验证整个流程、生成图表，无需下载城市路网
python main.py --region Demo --agents 200 --seeds 32 33 34 --plots --maps --output results/demo

# 默认网格：10 个种子 × 3 个压力 × 3 个灾害强度 × 6 个情景
python main.py --region Japan --plots --output results/tokyo
python main.py --region 'Hong Kong' --plots --output results/hong_kong

# 只重绘已完成实验，不重新下载路网或运行仿真
python main.py --plot-only results/demo --maps
```

城市首次运行需要网络。OSMnx 下载并缓存 GraphML；也可以 `--graph /path/to/network.graphml` 指定已有快照。输入必须是经纬度坐标、有向图，边有以秒为单位的 `travel_time`。本次执行环境和包版本随结果记录在 `manifest.json`。`requirements-lock.txt` 是本次验证环境的版本快照。

输出目录必须为空，防止不同模型版本覆盖彼此。旧 CSV、图片和日志移至 `legacy_outputs/`，**不能与新模型结果混用**。旧 Python 脚本备份在项目同级目录 `../nhb_before_refactor_20261010/`。

## 核心代码

### 容量感知推荐与 OSM 双地区实验

新增 `Adaptive` 情景，保留 Baseline、Full、Friction、Exposure、Fastest。它先生成患者无干预选择，再按出发时刻（同刻按 agent_id）处理推荐，依据此前患者已经承诺的到达时刻预测 FCFS 等待。只能查看过去决策，不能查看未来出发患者或最终实现的等待。预计总耗时至少减少 5 分钟、额外出行不超过 10 分钟且符合临床类型时，80% 的患者接受推荐；每次决策更新预约表。未来患者可能更早到达，最终排队统一按实际到达时间重算，因此推荐不保证每人获益。假设所有患者原始选择均可被平台观察、无背景流量、无取消，设施与出行时间信息准确；这属于理想信息条件。

参数为 `recommendation_min_gain_min`、`recommendation_max_detour_min`、`recommendation_compliance`，可通过 JSON 修改。正文结果只涵盖默认信息与遵从条件，不能声称对信息误差稳健。

本轮只用 OpenStreetMap 香港及东京 23 区驾驶路网，不使用 `realdata` 日本轨迹。患者在节点均匀抽样，医院表仍为未核实示例。预设网格：每地区 200/1000/2000 名患者、120 分钟出发窗口、压力 6、灾害强度 0/1/3、种子 32–36，每设置六情景。需求水平不是实测患者数量。每次运行保存路网及代码快照和输入哈希。

```bash
# region 依次为 Japan、'Hong Kong'；agents 依次为 200、1000、2000
python main.py --region Japan --agents 1000 --seeds 32 33 34 35 36 --stress 6 --severity 0 1 3 --output results/osm_japan_1000
python main.py --region 'Hong Kong' --agents 1000 --seeds 32 33 34 35 36 --stress 6 --severity 0 1 3 --output results/osm_hk_1000
# 完成六个目录后导出完整正文图及配对指标
python paper_results.py
```

导出见 `results/osm_paper/`：完整种子指标、全网格配对效应、PNG/SVG/PDF 图和来源记录。配对 bootstrap 区间仅代表五种子 Monte Carlo 波动；不可作为真实患者的实证置信区间。轻载、高载和负面结果全部报告。论文正式结论还需要服务能力校准、人口起点权重、信息误差、遵从率、更多种子和独立验证。

- `config.py`：实验设计、默认参数与校验。
- `environment.py`：医院数据、OSM 路网、灾害情景、反向最短路、离线沙盘。
- `decision.py`：无量纲效用、压力权重、设施可选集合。
- `simulation.py`：固定种子的人群、共同随机数、先到先服务的多服务台排队。
- `analysis.py`：重复实验指标、按种子配对的效应与 bootstrap 区间。
- `visual.py`：假设机制图、压力/灾害扫描图、设施需求图、配对效应图、可选路线图。
- `main.py`：唯一运行和重绘入口。
- `data/facilities.csv`：设施元数据唯一来源。
- `tests/`：行为、反事实和端到端回归测试。

重复的 `analysis_Final.py`、`data_analysis.py`、`plot_LR.py`、`road.py`、`redraw.py`、`spatial_draw.py` 和 `visual_map.py` 已退出核心目录；无关的 `graph TD.rb` 也移至备份。重新绘图统一使用 `--plot-only`。

## 模型定义与明确假设

### 1. 空间灾害

道路中点到灾害中心距离为 d，范围为 r：

```text
intensity = max(0, 1 - d/r)
t_damaged = t_baseline * exp(severity * intensity)
```

这里的 intensity 是无量纲的情景暴露，不是真实积水深度。中点近似不解析道路上每个位置的洪水，指数系数未校准。`severity=0` 是无灾害对照。可用 `--closure 0.8` 将暴露强度 ≥0.8 的道路移除；断路阈值和时间惩罚是两个独立假设。零灾害情景不移除道路。

每个灾害情景都从未损坏快照生成，避免惩罚累积。保留不连通路网，不能因为不可达而删掉患者。从医院沿反向有向图计算最短时间，将每个情景的全图搜索次数降低到不同医院节点数；所有对照组复用相同时间矩阵。

OSMnx 的时间是由道路长度和速度推算的自由流时间，不包括动态交通拥堵。缺失速度使用 50 km/h fallback，正式实验应根据当地数据校准。接口依据：[OSMnx 官方文档](https://osmnx.readthedocs.io/en/stable/user-reference.html)。

### 2. 压力与选择

```text
alpha_i = sigmoid(logistic_slope * (threshold_i - stress))
R_ij = time_reference_sec / (time_reference_sec + travel_time_ij)
H_ij = prestige_j * attraction_size_j / size_reference * R_ij**gamma * multiplier_j
U_ij = alpha_i * R_ij + (1-alpha_i) * H_ij + temperature * Gumbel_ij
```

选择可达且符合就医类型的最大效用设施；同分按设施名排序。`multiplier` 对医院使用 friction，对诊所使用 exposure。两个效用项均无量纲，尺度在所有情景固定，不按干预后结果重新归一化。`gamma=1` 只是距离折现参数，**不等价于完全理性**。`temperature=0` 给出确定性选择；正值加入可复现的随机效用误差。

压力是外生情景变量，与道路损坏分别扫描，因此不能自动解读为洪水导致心理压力的因果估计。logistic 关系是输入假设；画出的机制曲线不是从模型自己的输出“验证”出的行为规律。`alpha` 目前只表示个体偏好权重；最快出行不等于最优诊疗结局。

`requires_core=True` 的患者只能选核心医院，其他患者允许选两类设施。默认比例 0.2 是示例假设，不是患病率估计。默认两个类型使用相同服务规则，未建急诊优先级；重症优先、专科需求、分诊、救护车等需要额外数据与机制才能研究。

### 3. 显式排队

患者出发时间默认均匀分布在 120 分钟窗口内，到达时间为出发时间 + 路上时间。每家设施有 `servers` 个平行服务台、每位患者占用固定 `service_minutes`；按到达顺序服务，同到达时间按 agent_id 排序。

```text
service_start = max(arrival_time, earliest_free_server)
wait = service_start - arrival_time
access = travel_time + wait  # 从出发到开始服务，不包括本人的诊疗时长
```

初始队列为空，服务持续到所有可达患者开始并完成服务，不在出发窗口结束时截断。没有事后容量缩放，也没有对大医院设置更高惩罚系数。`--service-rate` 可独立改变服务速率，需求强度由人数和出发窗口决定。

Baseline、Full、Friction、Exposure 和 Fastest 是一次性选择后排队；Adaptive 在出发决策时依据此前已承诺的到达计划预测等待。各情景均没有途中改道、道路流量拥堵或灾后恢复。因此只能报告有限到达批次的服务延迟，不能据此宣称道路死锁解除、动态系统稳定或完整系统韧性。

## 实验与指标

六个情景共用每个种子的人群、起点、就医类型、出发时间和偏好随机项：

1. Baseline：无信息干预（仍处于该行设定的灾害情景）。
2. Full：医院降权 + 诊所曝光。
3. Friction：仅医院降权，强度与 Full 相同。
4. Exposure：仅诊所曝光，强度与 Full 相同。
5. Fastest：选择最快可达且符合类型的设施；仅为最短出行基准，不是系统最优策略。
6. Adaptive：从无干预选择出发，按出发顺序提供容量感知推荐，以预计出行与等待之和评估候选设施，并限制收益阈值、额外出行及遵从率。

同时输出交互效应 `Full - Friction - Exposure + Baseline`，不能简单把 Full 与某个单因素差异解释为交互。

主要报告：

- 全部患者的不可达比例、在 `access_target_min` 内开始服务的比例。
- 可达患者的平均出行时间、排队时间、开始服务时间及第 90 百分位。
- 超出最快合适设施的额外出行时间（描述偏离，不直接称为决策错误）。
- 设施需求 / 出发窗口内理论服务能力、该比值的变异系数、需求份额 HHI；包含零需求设施。需求能力比不是实测利用率，超过 1 也不自动证明死锁。
- 预先定义的低压力阈值组（阈值分布底部 25%）、需要核心医院组、可去诊所组。

时间指标对不可达者保留缺失值，不能将其填成 0 或随意设成 120 分钟；因此必须与不可达比例和全人群时限达标比例一起解读。同一灾害设定内各信息组的可达集合相同；不同灾害/关闭情景间的可达者均值不能直接做同一人群因果比较。

### 统计口径

先在每个种子内计算总体指标，再配对计算 treatment - baseline，最后对**种子层的差值**做 percentile bootstrap。正的时间差表示干预更慢；正的时限达标比例差表示干预更好。单一种子不产生区间。不输出患者级显著性检验，也不打印预设“显著改善”的句子。

区间仅描述固定设施、路网与参数下的人群/偏好 Monte Carlo 变异，不能代替实证置信区间。种子不是独立城市或独立灾害样本。正式报告应提高重复次数并检查收敛，对多个地区、灾害中心与参数作稳健性检验；默认 10 个种子用于开发验证，不保证论文统计精度。

## 数据与可复现性

### OSM 双地区验证

```bash
# region 依次为 Japan、'Hong Kong'；agents 依次为 200、1000、2000
python main.py --region Japan --agents 1000 --seeds 32 33 34 35 36 --stress 6 --severity 0 1 3 --output results/osm_japan_1000
python main.py --region 'Hong Kong' --agents 1000 --seeds 32 33 34 35 36 --stress 6 --severity 0 1 3 --output results/osm_hk_1000
python paper_results.py
```

这组实验使用 OpenStreetMap 驾驶路网，当前正文只报告香港和东京 23 区的仿真结果，不使用 `realdata` 日本地铁轨迹。患者在路网节点上均匀抽样，医院表仍为未核实示例。预设网格为每地区 200/1000/2000 名患者、120 分钟出发窗口、压力 6、灾害强度 0/1/3、种子 32–36，每个设置共六个情景。需求水平不是实测患者数量。

`results/osm_paper/` 保存完整种子指标、全网格配对效应、PNG/SVG/PDF 图和来源记录。配对 bootstrap 区间仅代表五种子 Monte Carlo 波动；不可作为真实患者的实证置信区间。轻载、高载和负面结果都应报告。正式研究仍需要服务能力校准、人口起点权重、信息误差、遵从率、更多种子和独立验证。

`data/facilities.csv` 沿用旧代码的名称、近似坐标、吸引力大小和声望参数，`source_status=illustrative_unverified`。这些值并未核实为真实医院容量或患者偏好。新增的服务台数（医院 6、诊所 2）与服务时长（10 分钟）同样是示例。`attraction_size` 只是效用代理变量，不能当作每小时服务能力；服务台与服务时长需独立估计。

正式研究需要补充带来源与日期的设施服务范围、到达过程、实际接诊速率、灾害交通时间及行为选择数据。用独立数据估计压力权重、声望、尺度和选择噪声，区分校准数据与验证数据。论文主张应限制为当前数据能够检验的机制。

默认起点在路网节点均匀抽样，这会偏向路口密集区域，不等于居民人口分布。可输入人口权重：

```bash
python main.py --region Japan --origins data/population_origins.csv --output results/weighted
```

该 CSV 需要 `node_id,weight`，节点必须在所用 GraphML 中。医院匹配距离超过 2 km 会报错，不静默删除。每次运行保存：

- `config.json`、`manifest.json`：参数、版本、时区明确的时间、源代码和输入 SHA256。
- `network.graphml`、`facilities.csv`：实际使用的路网及设施快照。
- `facilities_input.csv`：服务速率缩放前的设施；复跑相同配置时用它作为 `--facilities` 输入，避免重复缩放。
- `source/`：本次运行的核心代码快照；使用外部人口权重时也保存 `origins.csv`。
- `decisions.csv`：每位患者的配对原始结果。
- `run_summary.csv`、`facility_loads.csv`：每次重复的指标及全部设施负载。
- `paired_effects.csv`：各情景相对基线和交互效应、区间、有效种子数。
- PNG/SVG 图：没有挑选“最好看”的种子；路线图固定第一个配置种子，设施图固定最高灾害强度与中间压力。

## 参数稳健性运行

所有 `ExperimentConfig` 字段都能通过 JSON 设置，命令行覆盖同名选项。例如保存 `settings.json`：

```json
{
  "region": "Demo",
  "n_agents": 500,
  "seeds": [32, 33, 34, 35, 36],
  "stress_levels": [3.5, 6.0, 8.5],
  "severity_levels": [0, 3, 6],
  "departure_window_min": 120,
  "service_rate_multiplier": 1.0,
  "requires_core_fraction": 0.2,
  "temperature": 0.05
}
```

```bash
python main.py --config settings.json --gamma 1 --output results/gamma1
python main.py --config settings.json --temperature 0 --output results/deterministic
python main.py --config settings.json --service-rate 0.5 --output results/slow_service
python main.py --config settings.json --agents 1000 --output results/high_demand
python main.py --config settings.json --friction 0.5 --exposure 2 --output results/mild_nudge
python main.py --config settings.json --closure 0.8 --output results/closures
```

不同参数点分目录保存；应报告完整扫描范围，保留无效和负面效果。城市实验还应改变灾害中心，并评估跨地区结果是否一致。论文题目、数据和方法章节尚未提供，此结构是围绕现有代码推导的可检验研究版本，不声称已经匹配某篇论文全部要求。
