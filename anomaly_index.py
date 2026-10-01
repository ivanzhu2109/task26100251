"""进程内群体评分异常索引。

每个实例对应一个固定触点和统计期。调用串行执行，不持久化。

统计口径：
- 个人均分 = 该受访者全部评分的算术平均，四舍五入到小数点后 6 位
  （恰好半单位时向上取），之后的判定把该定点值视为精确值。
- 群体均值 = 所有受访者个人均分的等权平均；标准差为总体标准差（含本人）。
- |个人均分 - 群体均值| > 2 * 总体标准差（严格大于）判为异常；
  没有受访者或方差为零时没有异常。

实现要点（全程整数精确运算，无浮点容差）：
- 个人均分放大 1e6 倍后为整数，记为 x。维护 R（人数）、S（Σx）、Q（Σx²）。
- 异常判定 (R*x - S)² > 4*(R*Q - S²) 为纯整数比较，边界精确。
- 所有 (x, 序号) 键存放在随机 treap 中，异常者即落在两根整数阈值
  之外的键；每次更新只对阈值变化区间做范围枚举，输出量即翻转人数 d。
- add_rating 期望 O(log R + d)，summary O(1)，anomalies O(log R + k)，
  常驻空间 O(R)。
"""
from __future__ import annotations

import random
from collections import namedtuple
from math import isqrt
from typing import Any, Dict, Iterator, Optional, Set, Tuple

_SCALE = 1_000_000          # 个人均分放大倍数（6 位定点）
_INF_SEQ = 1 << 62          # 键范围上界使用的“无穷大”序号

Summary = namedtuple("Summary", ["respondents", "anomalies", "mean", "variance"])


def _round6(total: int, count: int) -> int:
    """total/count 四舍五入到 1e-6（半单位向上），返回放大 1e6 倍的整数。"""
    q, r = divmod(total * _SCALE, count)
    if 2 * r >= count:
        q += 1
    return q


class _Node:
    __slots__ = ("key", "prio", "rid", "left", "right")

    def __init__(self, key: Tuple[int, int], prio: int, rid: Any) -> None:
        self.key = key          # (mean_scaled, seq)
        self.prio = prio
        self.rid = rid
        self.left: Optional[_Node] = None
        self.right: Optional[_Node] = None


def _split(root: Optional[_Node], key: Tuple[int, int]):
    """分裂为 (< key, >= key) 两棵 treap。"""
    if root is None:
        return (None, None)
    if key <= root.key:
        left, right = _split(root.left, key)
        root.left = right
        return (left, root)
    left, right = _split(root.right, key)
    root.right = left
    return (root, right)


def _merge(a: Optional[_Node], b: Optional[_Node]) -> Optional[_Node]:
    if a is None:
        return b
    if b is None:
        return a
    if a.prio < b.prio:
        a.right = _merge(a.right, b)
        return a
    b.left = _merge(a, b.left)
    return b


def _iter_range(node: Optional[_Node], lo_key, hi_key) -> Iterator[_Node]:
    """升序产生键位于 [lo_key, hi_key] 的节点；None 表示该侧无界。"""
    if node is None:
        return
    if lo_key is not None and node.key < lo_key:
        yield from _iter_range(node.right, lo_key, hi_key)
    elif hi_key is not None and node.key > hi_key:
        yield from _iter_range(node.left, lo_key, hi_key)
    else:
        yield from _iter_range(node.left, lo_key, hi_key)
        yield node
        yield from _iter_range(node.right, lo_key, hi_key)


class GroupAnomalyIndex:
    """单个固定触点/统计期的受访者异常标记索引。"""

    def __init__(self) -> None:
        self._rec: Dict[Any, list] = {}   # rid -> [total, count, mean_scaled, seq]
        self._root: Optional[_Node] = None
        self._n = 0                       # 受访者数 R
        self._sum = 0                     # Σ mean_scaled
        self._sq = 0                      # Σ mean_scaled²
        self._anom = 0                    # 当前异常人数
        self._seq = 0
        self._rand = random.Random()

    # ---- 内部：阈值与判定 -------------------------------------------------

    def _thresholds(self) -> Tuple[Optional[int], Optional[int]]:
        """返回 (low_max, high_min)：均分 <= low_max 或 >= high_min 判为异常。

        无受访者或方差为零时返回 (None, None)，表示不存在异常。
        """
        n = self._n
        if n == 0:
            return (None, None)
        t = 4 * (self._sq * n - self._sum * self._sum)   # = 4R²σ²（放大 1e12）
        if t <= 0:
            return (None, None)
        s = isqrt(t)
        # x 为整数：(S - R*x) > sqrt(t)  <=>  x <= (S - isqrt(t) - 1) // R
        low_max = (self._sum - s - 1) // n
        # (R*x - S) > sqrt(t)  <=>  x >= ceil((S + isqrt(t) + 1) / R)
        high_min = (self._sum + s + n) // n
        return (low_max, high_min)

    @staticmethod
    def _outside(x: int, low_max: Optional[int], high_min: Optional[int]) -> bool:
        return (low_max is not None and x <= low_max) or (
            high_min is not None and x >= high_min
        )

    def _drain(self, a: Optional[int], b: Optional[int], out: Set[Any]) -> None:
        """把均分落在 [a, b]（None 表示无界）内的受访者 ID 加入 out。"""
        if a is not None and b is not None and a > b:
            return
        lo_key = (a, 0) if a is not None else None
        hi_key = (b, _INF_SEQ) if b is not None else None
        for node in _iter_range(self._root, lo_key, hi_key):
            out.add(node.rid)

    # ---- 公开接口 ---------------------------------------------------------

    def add_rating(self, respondent_id: Any, score: int) -> Tuple[Set[Any], Set[Any]]:
        """追加一条评分，返回 (进入异常集合, 解除异常集合)。"""
        if not isinstance(score, int) or isinstance(score, bool) or not 1 <= score <= 10:
            raise ValueError("score 必须是 1..10 的整数")

        rec = self._rec.get(respondent_id)
        low_old, high_old = self._thresholds()

        if rec is None:
            old_mean = None
            old_out = False
            self._seq += 1
            rec = [0, 0, 0, self._seq]
            self._rec[respondent_id] = rec
        else:
            old_mean = rec[2]
            old_out = self._outside(old_mean, low_old, high_old)

        rec[0] += score
        rec[1] += 1
        new_mean = _round6(rec[0], rec[1])

        if new_mean == old_mean:
            # 均分未变 => 群体统计量与阈值全部不变 => 无任何翻转
            return (set(), set())

        if old_mean is not None:
            # 从 treap 删除旧键
            left, rest = _split(self._root, (old_mean, rec[3]))
            _, right = _split(rest, (old_mean, rec[3] + 1))
            self._root = _merge(left, right)
            self._sum += new_mean - old_mean
            self._sq += new_mean * new_mean - old_mean * old_mean
        else:
            self._n += 1
            self._sum += new_mean
            self._sq += new_mean * new_mean
        rec[2] = new_mean

        node = _Node((new_mean, rec[3]), self._rand.getrandbits(64), respondent_id)
        left, right = _split(self._root, node.key)
        self._root = _merge(_merge(left, node), right)

        low_new, high_new = self._thresholds()
        new_out = self._outside(new_mean, low_new, high_new)

        # 对其余受访者（键未变）求翻转：entered = P_new ∧ ¬P_old，
        # exited = P_old ∧ ¬P_new，各自由低/高两侧均值区间精确给出。
        entered: Set[Any] = set()
        exited: Set[Any] = set()
        if low_new is not None:
            a = low_old + 1 if low_old is not None else None
            b = low_new if high_old is None else min(low_new, high_old - 1)
            self._drain(a, b, entered)
        if high_new is not None:
            a = high_new if low_old is None else max(high_new, low_old + 1)
            b = high_old - 1 if high_old is not None else None
            self._drain(a, b, entered)
        if low_old is not None:
            a = low_new + 1 if low_new is not None else None
            b = low_old if high_new is None else min(low_old, high_new - 1)
            self._drain(a, b, exited)
        if high_old is not None:
            a = high_old if low_new is None else max(high_old, low_new + 1)
            b = high_new - 1 if high_new is not None else None
            self._drain(a, b, exited)

        # 被修改者的旧键已不存在，范围枚举对其无效，单独按新旧状态修正。
        entered.discard(respondent_id)
        exited.discard(respondent_id)
        if new_out and not old_out:
            entered.add(respondent_id)
        elif old_out and not new_out:
            exited.add(respondent_id)

        self._anom += len(entered) - len(exited)
        return (entered, exited)

    def summary(self) -> Summary:
        """返回 Summary(respondents, anomalies, mean, variance)；数值为浮点近似。"""
        n = self._n
        if n == 0:
            return Summary(0, 0, 0.0, 0.0)
        mean = self._sum / (n * _SCALE)
        variance = (self._sq * n - self._sum * self._sum) / (n * n * _SCALE * _SCALE)
        return Summary(n, self._anom, mean, variance)

    def anomalies(self) -> Dict[Any, float]:
        """返回当前全部异常受访者及其个人均分（浮点近似）。"""
        low_max, high_min = self._thresholds()
        result: Dict[Any, float] = {}
        if low_max is not None:
            for node in _iter_range(self._root, None, (low_max, _INF_SEQ)):
                result[node.rid] = node.key[0] / _SCALE
        if high_min is not None:
            for node in _iter_range(self._root, (high_min, 0), None):
                result[node.rid] = node.key[0] / _SCALE
        return result
