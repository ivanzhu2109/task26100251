"""进程内群体评分异常索引（纯标准库实现）。

一个 AnomalyIndex 实例对应一个固定触点与统计期。
评分 1..10 整数；个人均分四舍五入到 6 位小数（半单位向上），
之后把该 6 位定点值视为精确值参与全部判定。

核心思路：异常集合是"当前键值 + 当前阈值"的纯函数（无滞后），
因此不显式维护异常名单，而是用一棵以 (定点均分, 内部uid) 为键的
笛卡尔树保存全部受访者，子树聚合 计数/和/平方和。
判定完全用整数精确完成：记 S1=Σk, S2=Σk², R=人数, N=S2·R−S1²，
则 |k−μ̄| > 2σ  ⟺  (k·R−S1)² > 4N（N>0 时），无任何浮点/开方误差。
阈值 μ̄±2σ 的无理数值用宽度 2/R 的有理包络夹住，差量枚举只在
包络区间内进行，候选之外的键不可能翻转，候选逐个用整数测试过滤。
"""

from __future__ import annotations

import random
from decimal import Decimal
from fractions import Fraction
from math import isqrt
from typing import Any, Dict, Iterator, NamedTuple, Optional, Set, Tuple

__all__ = ["AnomalyIndex"]

_SCALE = 10 ** 6


def round6(num: int, den: int) -> int:
    """round(num/den, 6) 的精确实现，半单位向上，返回放大 1e6 倍的整数。"""
    q, r = divmod(num * _SCALE, den)
    if 2 * r >= den:
        q += 1
    return q


class Summary(NamedTuple):
    """summary() 的返回：受访者数、异常人数、群体均值、总体方差（展示用浮点）。"""

    respondent_count: int
    anomaly_count: int
    mean: float
    variance: float


class _State:
    """一次群体统计快照：S1/S2/R/N 与阈值 μ̄±2σ 的有理包络（键的放大整数单位）。"""

    __slots__ = ("s1", "s2", "r", "n", "l1", "l2", "h1", "h2")

    def __init__(self, s1: int, s2: int, r: int) -> None:
        self.s1 = s1
        self.s2 = s2
        self.r = r
        n = s2 * r - s1 * s1
        self.n = n
        if n > 0:
            p = isqrt(n)  # p ≤ √N < p+1，σ=√N/R ∈ [p/R, (p+1)/R)
            two_p = 2 * p
            self.l1: Fraction | None = Fraction(s1 - two_p - 2, r)
            self.l2: Fraction | None = Fraction(s1 - two_p, r)
            self.h1: Fraction | None = Fraction(s1 + two_p, r)
            self.h2: Fraction | None = Fraction(s1 + two_p + 2, r)
        else:
            self.l1 = self.l2 = self.h1 = self.h2 = None

    def has_rays(self) -> bool:
        return self.n > 0

    def is_anomalous(self, key: int) -> bool:
        """精确整数判定：|k−μ̄| > 2σ ⟺ (k·R−S1)² > 4N（N=0 时无异常）。"""
        n = self.n
        if n == 0:
            return False
        d = key * self.r - self.s1
        return d * d > 4 * n


class _Node:
    __slots__ = ("key", "uid", "prio", "left", "right", "cnt", "s1", "s2")

    def __init__(self, key: int, uid: int, prio: int) -> None:
        self.key = key
        self.uid = uid
        self.prio = prio
        self.left: _Node | None = None
        self.right: _Node | None = None
        self.cnt = 1
        self.s1 = key
        self.s2 = key * key


def _pull(node: _Node) -> None:
    cnt, s1, s2 = 1, node.key, node.key * node.key
    if node.left is not None:
        cnt += node.left.cnt
        s1 += node.left.s1
        s2 += node.left.s2
    if node.right is not None:
        cnt += node.right.cnt
        s1 += node.right.s1
        s2 += node.right.s2
    node.cnt, node.s1, node.s2 = cnt, s1, s2


def _merge(a: _Node | None, b: _Node | None) -> _Node | None:
    if a is None:
        return b
    if b is None:
        return a
    if a.prio < b.prio:
        a.right = _merge(a.right, b)
        _pull(a)
        return a
    b.left = _merge(a, b.left)
    _pull(b)
    return b


class _Treap:
    """以 (定点个人均分, uid) 为键的笛卡尔树，子树聚合计数与一、二阶和。"""

    __slots__ = ("root", "size", "_rng")

    def __init__(self) -> None:
        self.root: _Node | None = None
        self.size = 0
        self._rng = random.Random()

    # ---- 修改 ----

    def insert(self, key: int, uid: int) -> None:
        self.root = self._insert(self.root, _Node(key, uid, self._rng.getrandbits(64)))
        self.size += 1

    def _insert(self, node: _Node | None, new: _Node) -> _Node:
        if node is None:
            return new
        if new.prio < node.prio:
            left, right = self._split(node, new.key, new.uid)
            new.left, new.right = left, right
            _pull(new)
            return new
        if (new.key, new.uid) < (node.key, node.uid):
            node.left = self._insert(node.left, new)
        else:
            node.right = self._insert(node.right, new)
        _pull(node)
        return node

    def _split(self, node: _Node | None, key: int, uid: int):
        """按 (key, uid) 分裂为 (<, >=) 两棵树。"""
        if node is None:
            return None, None
        if (key, uid) <= (node.key, node.uid):
            left, right = self._split(node.left, key, uid)
            node.left = right
            _pull(node)
            return left, node
        left, right = self._split(node.right, key, uid)
        node.right = left
        _pull(node)
        return node, right

    def erase(self, key: int, uid: int) -> None:
        self.root = self._erase(self.root, key, uid)
        self.size -= 1

    def _erase(self, node: _Node, key: int, uid: int) -> _Node | None:
        if node.key == key and node.uid == uid:
            return _merge(node.left, node.right)
        if (key, uid) < (node.key, node.uid):
            node.left = self._erase(node.left, key, uid)
        else:
            node.right = self._erase(node.right, key, uid)
        _pull(node)
        return node

    # ---- 计数查询（bound 为 Fraction，与整数键精确比较）----

    def count_le(self, bound: Fraction) -> int:
        return self._count_le(self.root, bound)

    def _count_le(self, node: _Node | None, bound: Fraction) -> int:
        if node is None:
            return 0
        if node.key <= bound:
            left_cnt = node.left.cnt if node.left is not None else 0
            return left_cnt + 1 + self._count_le(node.right, bound)
        return self._count_le(node.left, bound)

    def count_ge(self, bound: Fraction) -> int:
        return self._count_ge(self.root, bound)

    def _count_ge(self, node: _Node | None, bound: Fraction) -> int:
        if node is None:
            return 0
        if node.key >= bound:
            right_cnt = node.right.cnt if node.right is not None else 0
            return right_cnt + 1 + self._count_ge(node.left, bound)
        return self._count_ge(node.right, bound)

    # ---- 范围枚举：O(log R + k) ----

    def iter_between(self, lo: Fraction, hi: Fraction) -> Iterator[tuple[int, int]]:
        """产出 lo ≤ key ≤ hi 的 (key, uid)。"""
        yield from self._iter_between(self.root, lo, hi)

    def _iter_between(self, node, lo, hi):
        if node is None:
            return
        if node.key >= lo:
            yield from self._iter_between(node.left, lo, hi)
        if lo <= node.key <= hi:
            yield node.key, node.uid
        if node.key <= hi:
            yield from self._iter_between(node.right, lo, hi)

    def iter_le(self, bound: Fraction) -> Iterator[tuple[int, int]]:
        yield from self._iter_le(self.root, bound)

    def _iter_le(self, node, bound):
        if node is None:
            return
        if node.key <= bound:
            yield from self._iter_le(node.left, bound)
            yield node.key, node.uid
            yield from self._iter_le(node.right, bound)
        else:
            yield from self._iter_le(node.left, bound)

    def iter_ge(self, bound: Fraction) -> Iterator[tuple[int, int]]:
        yield from self._iter_ge(self.root, bound)

    def _iter_ge(self, node, bound):
        if node is None:
            return
        if node.key >= bound:
            yield from self._iter_ge(node.left, bound)
            yield node.key, node.uid
            yield from self._iter_ge(node.right, bound)
        else:
            yield from self._iter_ge(node.right, bound)


class AnomalyIndex:
    """群体评分异常索引。

    add_rating(respondent_id, score) -> (entered, exited) 两个集合；
    summary() -> Summary(受访者数, 异常人数, 群体均值, 总体方差)；
    anomalies() -> {respondent_id: 个人均分(Decimal)}。

    复杂度（R=受访者数，d=本次翻转人数，k=查询返回人数）：
    add_rating 期望 O(log R + d)，summary O(log R)，anomalies O(log R + k)，
    常驻空间 O(R)。
    """

    def __init__(self) -> None:
        self._resp: Dict[Any, list] = {}  # id -> [score_sum, rating_cnt, uid]
        self._ids: list = []              # uid -> id
        self._tree = _Treap()
        self._prev: _State | None = None  # 上一次 add_rating 后的群体统计快照

    # ---- 内部 ----

    def _cur_state(self) -> _State | None:
        root = self._tree.root
        if root is None:
            return None
        return _State(root.s1, root.s2, root.cnt)

    def _candidates(self, prev: _State | None, cur: _State) -> Iterator[tuple[int, int]]:
        """枚举所有可能翻转的 (key, uid)：新旧阈值包络之间（或空集→射线）。"""
        prev_rays = prev is not None and prev.has_rays()
        cur_rays = cur.has_rays()
        if prev_rays and cur_rays:
            yield from self._tree.iter_between(min(prev.l1, cur.l1), max(prev.l2, cur.l2))
            yield from self._tree.iter_between(min(prev.h1, cur.h1), max(prev.h2, cur.h2))
        elif cur_rays:
            yield from self._tree.iter_le(cur.l2)
            yield from self._tree.iter_ge(cur.h1)
        elif prev_rays:
            yield from self._tree.iter_le(prev.l2)
            yield from self._tree.iter_ge(prev.h1)

    # ---- 公开接口 ----

    def add_rating(self, respondent_id, score: int) -> Tuple[Set, Set]:
        """追加一条评分，返回 (本次进入异常的集合, 本次解除异常的集合)。"""
        if not isinstance(score, int) or not 1 <= score <= 10:
            raise ValueError(f"score 必须是 1..10 的整数，收到: {score!r}")
        rec = self._resp.get(respondent_id)
        if rec is None:
            rec = [0, 0, len(self._ids)]
            self._ids.append(respondent_id)
            self._resp[respondent_id] = rec
            old_key = None
        else:
            old_key = round6(rec[0], rec[1])
        rec[0] += score
        rec[1] += 1
        new_key = round6(rec[0], rec[1])
        if old_key == new_key:
            # 个人均分未变 => 群体统计量不变 => 异常集合不变
            return set(), set()
        uid = rec[2]
        if old_key is not None:
            self._tree.erase(old_key, uid)
        self._tree.insert(new_key, uid)
        prev = self._prev
        cur = self._cur_state()
        self._prev = cur

        entered: Set = set()
        exited: Set = set()
        # 被修改者本人直接判定
        old_s = old_key is not None and prev is not None and prev.is_anomalous(old_key)
        new_s = cur.is_anomalous(new_key)
        if new_s and not old_s:
            entered.add(respondent_id)
        elif old_s and not new_s:
            exited.add(respondent_id)
        # 其他人：只有落在新旧阈值包络之间的键才可能翻转
        for key, cuid in self._candidates(prev, cur):
            if cuid == uid:
                continue
            o = prev is not None and prev.is_anomalous(key)
            n = cur.is_anomalous(key)
            if n and not o:
                entered.add(self._ids[cuid])
            elif o and not n:
                exited.add(self._ids[cuid])
        return entered, exited

    def summary(self) -> Summary:
        """(受访者数, 异常人数, 群体均值, 总体方差)；数值为展示用浮点近似。"""
        st = self._cur_state()
        if st is None:
            return Summary(0, 0, 0.0, 0.0)
        r = st.r
        anomaly_count = 0
        if st.has_rays():
            # 低侧精确异常数 = (key ≤ L2 的个数) − 包络 [L1, L2] 内非异常的个数
            low = self._tree.count_le(st.l2)
            for key, _ in self._tree.iter_between(st.l1, st.l2):
                if not st.is_anomalous(key):
                    low -= 1
            high = self._tree.count_ge(st.h1)
            for key, _ in self._tree.iter_between(st.h1, st.h2):
                if not st.is_anomalous(key):
                    high -= 1
            anomaly_count = low + high
        mean = st.s1 / (r * _SCALE)
        variance = st.n / (r * r * _SCALE * _SCALE)
        return Summary(r, anomaly_count, mean, variance)

    def anomalies(self) -> Dict[Any, Decimal]:
        """当前全部异常受访者及其个人均分（Decimal，精确的 6 位定点值）。"""
        st = self._cur_state()
        if st is None or not st.has_rays():
            return {}
        out: Dict[Any, Decimal] = {}
        for key, uid in self._tree.iter_le(st.l2):
            if st.is_anomalous(key):
                out[self._ids[uid]] = Decimal(key).scaleb(-6)
        for key, uid in self._tree.iter_ge(st.h1):
            if st.is_anomalous(key):
                out[self._ids[uid]] = Decimal(key).scaleb(-6)
        return out
