# 群体评分异常索引

纯 Python 标准库实现的进程内群体评分异常索引，用于反馈平台实时维护受访者异常标记。
每个 `AnomalyIndex` 实例对应一个固定触点与统计期；调用串行执行，不做持久化。

## 统计口径

- 评分为 1..10 整数，归属一个受访者；个人均分先四舍五入到小数点后 6 位
  （半单位向上），之后该 6 位定点值被视为精确值。
- 群体均值 = 所有受访者个人均分的等权平均；标准差为总体标准差（含被判断者本人）。
- 个人均分与群体均值距离**严格大于** 2 倍总体标准差才算异常；无受访者或方差为零时无异常。

## 使用

```python
from anomaly_index import AnomalyIndex

idx = AnomalyIndex()
entered, exited = idx.add_rating("resp-1", 10)   # 追加评分，返回 (进入异常, 解除异常) 两个集合
s = idx.summary()        # Summary(respondent_count, anomaly_count, mean, variance)，浮点近似
idx.anomalies()          # {respondent_id: 个人均分(Decimal)}，当前全部异常者
```

## 实现要点

- 异常集合是"当前键值 + 当前阈值"的纯函数，不显式维护名单；一棵以
  `(定点均分, 内部uid)` 为键的笛卡尔树保存全部受访者，子树聚合计数/和/平方和。
- 判定完全为整数精确运算：记 `S1=Σk, S2=Σk², R=人数, N=S2·R−S1²`，则
  `|k−μ̄| > 2σ ⟺ (k·R−S1)² > 4N`（N>0 时），无浮点容差、不开方。
- 无理数阈值 `μ̄±2σ` 用宽度 `2/R` 的有理包络夹住；每次投稿只枚举新旧包络之间的
  候选键（即真正翻转者加 O(1) 个边界候选），逐个用整数测试过滤得到差量。
- 复杂度（R=受访者数，d=本次翻转人数，k=查询返回人数）：
  `add_rating` 期望 `O(log R + d)`，`summary` `O(log R)`，`anomalies` `O(log R + k)`，
  常驻空间 `O(R)`，不保存逐条历史。

## 文件

- `anomaly_index.py` — 索引实现（含笛卡尔树）。
- `test_anomaly_index.py` — 单元测试：边界场景 + 随机投稿流对照独立全量重算参考实现。
- `scale_test.py` — 规模用例：20 万受访者 + 追加 100 万条评分，常驻异常者 ≥ 1 万。

## 运行

```bash
python3 -m unittest test_anomaly_index -v   # 普通测试（也可用 pytest）
python3 scale_test.py                       # 规模用例（约 1-2 分钟）
```
