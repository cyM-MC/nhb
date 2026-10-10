# OSM 双地区机制仿真：正文结果草稿

## 方法

We evaluated a capacity-aware recommendation policy on OpenStreetMap driving networks for Hong Kong and Tokyo's 23 wards. The policy retained clinical eligibility constraints and processed patients in departure-time order. At each decision, it estimated waiting time using only previously committed arrivals and recommended an alternative only when the predicted reduction in travel-plus-waiting time was at least five minutes and additional travel did not exceed ten minutes. Acceptance was set to 80%. Realized queues were reconstructed from arrival order after all decisions, allowing later departures to overtake earlier ones. Thus, recommendations used no future patient decisions and did not guarantee individual realized benefit.

The full factorial simulation crossed two networks, three demand levels (200, 1,000 and 2,000 patients departing over 120 minutes), three hypothetical hazard severities (0, 1 and 3), and six policies, with five paired population seeds per setting. Stress was fixed at six. Policies shared populations and preference noise. The six policies were no intervention, fixed combined intervention, hospital downweighting, clinic exposure, fastest-travel choice, and capacity-aware recommendation. Facility attributes, service capacities, behavioural parameters and hazard intensities were illustrative assumptions, not empirical estimates. Origins were sampled uniformly over road nodes. The current results section reports only the Hong Kong and Japan OSM road-network experiments; the Japanese railway data branch was not used in this round of validation.

## 结果

Under 1,000-patient demand and severity three, capacity-aware recommendations reduced mean departure-to-service-start time from 83.65 to 10.12 minutes in Hong Kong (paired difference -73.53 minutes; five-seed percentile bootstrap interval -79.42 to -66.05) and from 233.56 to 44.64 minutes in Tokyo (difference -188.93; interval -199.47 to -172.83). The fraction starting service within 60 minutes increased from 55.12% to 97.70% in Hong Kong and from 40.58% to 73.90% in Tokyo.

The reductions primarily reflected lower waiting times. Mean waiting decreased from 74.42 to 3.28 minutes in Hong Kong and from 218.92 to 29.78 minutes in Tokyo. Tokyo travel time increased slightly, from 14.64 to 14.86 minutes, illustrating a travel-versus-service-delay trade-off rather than universal travel improvement. Unreachable shares were unchanged (2.28% and 0.48%, respectively), because recommendations cannot restore disconnected routes.

Policy performance depended on demand and geography. At low Hong Kong demand (200 patients) and severity three, the fixed combined intervention increased mean access time from 9.55 to 11.71 minutes, whereas the adaptive policy achieved 6.69 minutes. At the same setting, fastest-travel choice was 5.63 minutes but could concentrate demand enough to worsen queueing in larger-demand runs. The same pattern appeared in Tokyo: at 1,000 patients and severity three, Baseline averaged 233.56 minutes, Fastest 176.76 minutes, and Adaptive 44.64 minutes, so the recommendation layer helped most when congestion pressure was high. These comparisons support considering both travel and service congestion rather than indiscriminate facility-type promotion or travel time alone within this model.

## 图注

**Figure: Conditional effects of information policies across demand and hypothetical hazard severity.** Each panel reports intervention-minus-baseline mean departure-to-service-start time for one network and demand level. Negative values indicate lower access time. Lines show means of five paired seed differences; shaded bands show 95% percentile bootstrap intervals over those differences. The compared policies are fixed combined intervention (Full), shortest-travel selection (Fastest), and capacity-aware recommendation (Adaptive). Hazard severity is a dimensionless scenario coefficient, not a measured flood depth or observed disaster category. Demand is simulated, and facility capacities are illustrative. Bands quantify Monte Carlo variability conditional on the fixed network and assumptions, not empirical uncertainty.

## 使用限制

这些段落可作为正文仿真结果初稿，不能写成实证医疗有效性结论。大幅收益依赖基线集中需求、示例服务能力和准确预约信息；没有背景流量、临床优先级、取消、动态路况或容量损失。仅五个种子、不扫描压力/信息误差/遵从率，尚不足以支撑 NHB 投稿的完整稳健性论证。正文需要同时呈现全部网格和负面结果，不只呈现上述示例。正式研究应校准容量与人口起点，并检验信息延迟、预测误差、遵从率、不同灾害中心和更多随机重复。

图：`results/osm_paper/access_effects.pdf`（矢量），同名 SVG/PNG。完整指标：`main_effects.csv` 和 `all_run_summary.csv`。来源及输入快照：六个 `results/osm_*` 目录和汇总 `provenance.json`。路网来源与时间推算方法参考 [OSMnx 官方文档](https://osmnx.readthedocs.io/en/stable/user-reference.html)；OSM 数据应按 [OpenStreetMap attribution guidelines](https://www.openstreetmap.org/copyright) 在最终论文中注明。
