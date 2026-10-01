# 群体评分异常索引

进程内、纯标准库实现的受访者异常标记索引（`anomaly_index.py`）。
每个 `GroupAnomalyIndex` 实例对应一个固定触点和统计期；调用串行执行，
不持久化、不保存逐条历史，常驻空间 O(R)。

## 统计口径

- 个人均分：受访者全部评分的均值，四舍五入到小数点后 6 位（半单位向上），
  之后判定把该定点值视为精确值（实现中放大 1e6 倍以整数精确运算）。
- 群体均值：所有受访者个人均分的等权平均；标准差为总体标准差（含本人）。
- 异常：`|个人均分 - 群体均值| > 2 * 总体标准差`（严格大于）；
  无受访者或方差为零时无异常。

## 接口

```python
from anomaly_index import GroupAnomalyIndex

idx = GroupAnomalyIndex()
entered, exited = idx.add_rating("resp-1", 7)   # 进入/解除异常的 ID 集合
s = idx.summary()        # Summary(respondents, anomalies, mean, variance)
idx.anomalies()          # {respondent_id: 个人均分}
```

- `add_rating`：期望 O(log R + d)，d 为本次实际翻转人数；返回的两个集合
  分别等于更新前后完整异常集合的增加项与减少项（含未被本次修改的他人）。
- `summary`：O(1)，数值为浮点近似。
- `anomalies`：O(log R + k)，k 为返回人数。

实现：随机 treap 维护 `(均分, 序号)` 有序键，异常者即两根整数阈值之外的键；
每次更新只对阈值变化区间做范围枚举，输出量即翻转人数，不扫描全员、
不枚举全部异常者。

## 运行

```bash
python3 -m unittest test_anomaly_index.py -v   # 普通测试（边界场景 + 随机对拍）
python3 scale_test.py                          # 规模用例：20 万受访者 + 100 万条评分
python3 scale_test.py quick                    # 缩小版规模自检
```
