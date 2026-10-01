"""AnomalyIndex 自动化测试：边界场景 + 随机流对照独立参考实现。

运行：python3 -m unittest test_anomaly_index -v   （或 python3 -m pytest）
"""

import math
import random
import unittest
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction

from anomaly_index import AnomalyIndex


def ref_round6(score_sum: int, cnt: int) -> Fraction:
    """独立的参考舍入：Decimal 四舍五入（半单位向上）到 6 位，返回精确分数。"""
    q = (Decimal(score_sum) / Decimal(cnt)).quantize(
        Decimal("0.000001"), rounding=ROUND_HALF_UP
    )
    return Fraction(q)


class Reference:
    """全量重算的参考实现：每次投稿后 O(R) 重算全部判定，完全独立于被测实现。"""

    def __init__(self):
        self.resp = {}  # id -> [sum, cnt]
        self.anom = set()
        self.keys = {}
        self.mean = 0.0
        self.var = 0.0

    def add(self, rid, score):
        rec = self.resp.setdefault(rid, [0, 0])
        rec[0] += score
        rec[1] += 1
        self.keys = {i: ref_round6(s, c) for i, (s, c) in self.resp.items()}
        r = len(self.keys)
        s1 = sum(self.keys.values())
        s2 = sum(k * k for k in self.keys.values())
        mean = Fraction(s1, r)
        var = Fraction(s2, r) - mean * mean  # 总体方差（精确）
        new_anom = {
            i for i, k in self.keys.items()
            if var > 0 and (k - mean) ** 2 > 4 * var
        }
        entered, exited = new_anom - self.anom, self.anom - new_anom
        self.anom = new_anom
        self.mean, self.var = float(mean), float(var)
        return entered, exited


def check_against_reference(testcase, idx, ref):
    """核对摘要、完整异常集合及个人均分。"""
    s = idx.summary()
    testcase.assertEqual(s.respondent_count, len(ref.resp))
    testcase.assertEqual(s.anomaly_count, len(ref.anom))
    testcase.assertTrue(math.isclose(s.mean, ref.mean, rel_tol=1e-12, abs_tol=1e-12))
    testcase.assertTrue(math.isclose(s.variance, ref.var, rel_tol=1e-12, abs_tol=1e-12))
    ano = idx.anomalies()
    testcase.assertEqual(set(ano), ref.anom)
    for rid, m in ano.items():
        k = ref.keys[rid]
        testcase.assertEqual(m, Decimal(k.numerator) / Decimal(k.denominator))


class TestBoundaryScenario(unittest.TestCase):
    """规格给定场景：10 恰在 2σ 边界不算异常；加一名 5 分者后进入；再加一名 10 分者后解除。"""

    def test_boundary_flip_flop(self):
        idx, ref = AnomalyIndex(), Reference()

        def step(rid, score):
            e1, x1 = idx.add_rating(rid, score)
            e2, x2 = ref.add(rid, score)
            self.assertEqual(e1, e2)  # 差量：进入集合
            self.assertEqual(x1, x2)  # 差量：解除集合
            check_against_reference(self, idx, ref)  # 完整集合 + 摘要
            return e1, x1

        for i in range(4):
            e, x = step(f"low{i}", 5)
            self.assertEqual((e, x), (set(), set()))
        e, x = step("high", 10)
        self.assertEqual((e, x), (set(), set()))  # 恰在边界，不是异常
        self.assertEqual(idx.anomalies(), {})

        e, x = step("low4", 5)
        self.assertEqual((e, x), ({"high"}, set()))  # 原来的 10 分者进入异常
        self.assertEqual(set(idx.anomalies()), {"high"})

        e, x = step("high2", 10)
        self.assertEqual((e, x), (set(), {"high"}))  # 原来的异常解除
        self.assertEqual(idx.anomalies(), {})

    def test_still_anomalous_not_re_entered(self):
        """仍然异常但均分改变的人不应重复进入。"""
        idx = AnomalyIndex()
        for i in range(7):
            idx.add_rating(f"n{i}", 5)
        e, x = idx.add_rating("out", 10)
        self.assertEqual((e, x), ({"out"}, set()))
        self.assertEqual(set(idx.anomalies()), {"out"})
        e, x = idx.add_rating("out", 9)  # 均分变为 9.5，仍异常 => 不得重复进入
        self.assertEqual((e, x), (set(), set()))
        self.assertEqual(idx.anomalies()["out"], Decimal("9.5"))


class TestRandomStreams(unittest.TestCase):
    """小规模随机投稿流，每次投稿后与全量重算参考实现核对。"""

    def run_stream(self, seed, n_ops, pick_rid, pick_score):
        rng = random.Random(seed)
        idx, ref = AnomalyIndex(), Reference()
        for _ in range(n_ops):
            rid = pick_rid(rng)
            score = pick_score(rng, rid)
            e1, x1 = idx.add_rating(rid, score)
            e2, x2 = ref.add(rid, score)
            self.assertEqual(e1, e2, f"entered mismatch: rid={rid} score={score}")
            self.assertEqual(x1, x2, f"exited mismatch: rid={rid} score={score}")
            check_against_reference(self, idx, ref)
        return idx, ref

    def test_uniform_small_group(self):
        # 人数少、分布均匀 => 阈值频繁移动，两侧都可能出现异常
        self.run_stream(
            seed=1, n_ops=1500,
            pick_rid=lambda rng: f"r{rng.randrange(12)}",
            pick_score=lambda rng, rid: rng.randint(1, 10),
        )

    def test_unbalanced_submission_counts(self):
        # 投稿次数极不均衡：两个重度用户占 80% 投稿
        heavy = ["heavy0", "heavy1"]

        def pick_rid(rng):
            if rng.random() < 0.8:
                return rng.choice(heavy)
            return f"r{rng.randrange(30)}"

        self.run_stream(
            seed=2, n_ops=2000, pick_rid=pick_rid,
            pick_score=lambda rng, rid: rng.randint(1, 10),
        )

    def test_two_sided_persistent_anomalies(self):
        # 低分与高分同时常驻异常
        def pick_score(rng, rid):
            if rid.startswith("low"):
                return rng.randint(1, 2)
            if rid.startswith("high"):
                return rng.randint(9, 10)
            return rng.randint(4, 6)

        def pick_rid(rng):
            p = rng.random()
            if p < 0.1:
                return f"low{rng.randrange(3)}"
            if p < 0.2:
                return f"high{rng.randrange(3)}"
            return f"mid{rng.randrange(20)}"

        idx, ref = self.run_stream(
            seed=3, n_ops=2000, pick_rid=pick_rid, pick_score=pick_score
        )
        self.assertGreater(len(ref.anom), 0)  # 确实存在常驻异常

    def test_rounding_boundaries(self):
        # 个人均分舍入边界：128 篇投稿使第 7 位小数恰为 5（半单位向上）；
        # 以及 3/6/7 等除不尽的分母。
        idx, ref = AnomalyIndex(), Reference()
        rng = random.Random(4)

        def step(rid, score):
            e1, x1 = idx.add_rating(rid, score)
            e2, x2 = ref.add(rid, score)
            self.assertEqual((e1, x1), (e2, x2))
            check_against_reference(self, idx, ref)

        # 127 个 1 分 + 1 个 2 分 => 均分 129/128 = 1.0078125 恰为半单位
        for _ in range(127):
            step("edge", 1)
        step("edge", 2)
        self.assertEqual(ref.keys["edge"], Fraction(1007813, 10**6))
        # 1/3、2/3、1/6、1/7 等舍入
        for score in (1, 2, 1, 2, 1, 2):
            step("thirds", score)
        self.assertEqual(ref.keys["thirds"], Fraction(15, 10))  # 1.5 精确
        step("seventh", 1)
        step("seventh", 2)
        # 混入随机流，让舍入边界值参与群体判定
        for _ in range(800):
            step(f"r{rng.randrange(15)}", rng.randint(1, 10))

    def test_zero_variance_regime(self):
        # 全员同分 => 方差为零无异常；随后被打破
        idx, ref = AnomalyIndex(), Reference()
        rng = random.Random(5)
        for i in range(20):
            e1, x1 = idx.add_rating(f"r{i}", 5)
            e2, x2 = ref.add(f"r{i}", 5)
            self.assertEqual((e1, x1), (e2, x2))
            check_against_reference(self, idx, ref)
        self.assertEqual(idx.summary().anomaly_count, 0)
        for _ in range(600):
            rid = f"r{rng.randrange(20)}"
            score = rng.randint(1, 10)
            e1, x1 = idx.add_rating(rid, score)
            e2, x2 = ref.add(rid, score)
            self.assertEqual((e1, x1), (e2, x2))
            check_against_reference(self, idx, ref)

    def test_fresh_index_and_single_respondent(self):
        idx = AnomalyIndex()
        self.assertEqual(idx.summary().respondent_count, 0)
        self.assertEqual(idx.summary().anomaly_count, 0)
        self.assertEqual(idx.anomalies(), {})
        e, x = idx.add_rating("solo", 1)
        self.assertEqual((e, x), (set(), set()))  # 单人无异常
        self.assertEqual(idx.anomalies(), {})
        with self.assertRaises(ValueError):
            idx.add_rating("solo", 0)
        with self.assertRaises(ValueError):
            idx.add_rating("solo", 11)

    def test_variance_collapse_to_zero(self):
        # 1 个 1 分 + 5 个 9 分 => 1 分者异常；随后全员收敛到 5.0 => 方差归零、异常清空
        idx, ref = AnomalyIndex(), Reference()

        def step(rid, score):
            e1, x1 = idx.add_rating(rid, score)
            e2, x2 = ref.add(rid, score)
            self.assertEqual((e1, x1), (e2, x2))
            check_against_reference(self, idx, ref)

        step("odd", 1)
        for i in range(5):
            step(f"n{i}", 9)
        self.assertEqual(set(idx.anomalies()), {"odd"})
        step("odd", 9)  # odd: (1+9)/2 = 5.0
        for i in range(5):
            for s in (1, 5, 5):  # (9+1+5+5)/4 = 5.0
                step(f"n{i}", s)
        self.assertEqual(ref.var, 0.0)
        self.assertEqual(idx.anomalies(), {})
        self.assertEqual(idx.summary().anomaly_count, 0)


if __name__ == "__main__":
    unittest.main()
