# 自来水厂水质预测与风险评估

基于 CSTR-LSTM 灰箱混合模型，集成特征筛选、动态时滞辨识与风险分级。

## 问题

1. **特征筛选与预测** — LASSO/RF/MI 三法交叉筛选 + Ridge 回归，预测出厂水浊度
2. **动态时滞辨识** — CCF 互相关 + Almon 多项式分布滞后模型，确定工艺变量时滞参数
3. **混合动态预测** — CSTR 串联 RTD 解析解 + Seq2Seq LSTM 残差修正，1-12h 超前预测
4. **风险分级评价** — S×D 双维积分 + K-Means 聚类，四级风险分类

## 运行

```bash
pip install -r requirements.txt
python q1_feature_prediction.py
python q2_lag_model.py
python q3_hybrid_forecast.py
python q4_risk_assessment.py
python validate.py
```

输出结果在 `output/` 目录。

## 文件结构

```
├── q1_feature_prediction.py   # 特征筛选 + 预测模型
├── q2_lag_model.py            # 时滞辨识 + Almon DLM
├── q3_hybrid_forecast.py      # CSTR + LSTM 混合预测
├── q4_risk_assessment.py      # 风险分级评价
├── validate.py                # 约束验证 + 灵敏度 + 鲁棒性
├── main.tex                   # 论文源文件
├── cumcmthesis.cls            # LaTeX 模板
├── figures/                   # 论文图表
├── output/                    # 计算结果
└── requirements.txt
```
