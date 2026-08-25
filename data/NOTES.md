# figs/data 数据快照（更新 2026-08-21 凌晨；源 results/RealRef/）

- avalign_offset_calib_v2.json：峰时间戳平移 15 偏移双片长（Fig.2a 主数据）
- null_model_mc.json：精确 MC 零模型（一对一匹配器，B=200）：chance 0.211，
  r=0.935，κ=−0.03±0.02，14/14 obs≤chance
- avalign_aac_path.json：容器通路直测 0.188→0.105，坍塌条目 ref05/06/12/13
- peaks_cache/ ×14：光流峰+onset 峰时间戳（一切离线复算的基础）
- ref*_desync_calib.json ×14（B.1 7 偏移，curves 含 argmax/expect 逐窗）
- nscaling_realref.json：σ(L) 0.072/0.056/0.043/0.034 @4/8/15/30s（60s 无 within-sd）
- clicktrain_xcorr.json + ref*_desync_calib_ext.json ×14（17 偏移含亚栅格/饱和）：
  跑完后拉取 → Fig.1 两 panel
- E2r 生成侧 chance（服务器现算，未存本地）：解析 0.222/0.230 vs 实测 0.223 → §5
- 匹配器注意：一切 IoU 复算必须用**一对一**匹配（third_party/TempoTokens 版）；
  javisbench 副本是多对一，同峰集读数 +50%（0.283），此差异已入文为实现漂移证据
