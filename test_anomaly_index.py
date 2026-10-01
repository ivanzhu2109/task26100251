"""GroupAnomalyIndex 自动化测试。

包含：
- 题目指定的 2 倍标准差边界场景（逐次核对差量与完整集合）；
- 与独立全量重算参考实现（Fraction 精确运算）对拍的随机投稿流，
  覆盖投稿次数极不均衡、低分与高分同时异常、个人均分舍入边界；
- 舍入、空库、零方差、非法输入等单元测试。
"""
import random
import unittest
from fractions import Fraction

from anomaly_index import GroupAnomalyIndex, Summary, _round6

_SCALE = 1_000_000


def _ref_round6(total: int, count: int) -> int:
    """独立的舍入参考：Fraction 精确判断半单位向上。"""
    f = Fraction(total * _SCALE, count)
    q = f.numerator // f.denominator
    if f - q >= Fraction(1, 2):
        q += 1
    return q


class RefModel:
    """全量重算参考实现：每次操作后从原始记录完整重算，O(R) 扫描。"""

    def __init__(self):
        self.data = {}  # rid -> [total, count]

    def _means(self):
        return {
            rid: Fraction(_ref_round6(t, c), _SCALE)
            for rid, (t, c) in self.data.items()
        }

    def anomaly_set(self):
        means = self._means()
        n = len(means)
        if n == 0:
            return {}
        mu = sum(means.values()) / n
        var = sum((m - mu) ** 2 for m in means.values()) / n
        if var == 0:
            return {}
        return {rid: m for rid, m in means.items() if (m - mu) ** 2 > 4 * var}

    def add_rating(self, rid, score):
        before = self.anomaly_set()
        self.data.setdefault(rid, [0, 0])
        self.data[rid][0] += score
        self.data[rid][1] += 1
        after = self.anomaly_set()
        return set(after) - set(before), set(before) - set(after)

    def summary(self):
        means = self._means()
        n = len(means)
        if n == 0:
            return 0, 0, 0.0, 0.0
        mu = sum(means.values()) / n
        var = sum((m - mu) ** 2 for m in means.values()) / n
        return n, len(self.anomaly_set()), float(mu), float(var)


class BoundaryScenarioTest(unittest.TestCase):
    """4 人均分 5、1 人 10：10 恰在 2σ 边界不算异常；加一名 5 分者后
    10 分者进入异常；再加一名 10 分者后异常解除。"""

    def test_scenario(self):
        idx, ref = GroupAnomalyIndex(), RefModel()
        steps = [("a", 5), ("b", 5), ("c", 5), ("d", 5), ("t", 10),
                 ("e", 5), ("t2", 10)]
        expected = [
            (set(), set()), (set(), set()), (set(), set()), (set(), set()),
            (set(), set()),          # 10 恰在边界，不是异常
            ({"t"}, set()),          # 边界收紧，t 进入异常
            (set(), {"t"}),          # 第二个 10 分者加入，t 解除异常
        ]
        for (rid, score), (exp_in, exp_out) in zip(steps, expected):
            got_in, got_out = idx.add_rating(rid, score)
            ref_in, ref_out = ref.add_rating(rid, score)
            self.assertEqual(got_in, exp_in, f"after {rid}={score} entered")
            self.assertEqual(got_out, exp_out, f"after {rid}={score} exited")
            self.assertEqual(got_in, ref_in)
            self.assertEqual(got_out, ref_out)
            # 完整异常集合核对
            self.assertEqual(set(idx.anomalies()), set(ref.anomaly_set()))
        s = idx.summary()
        self.assertEqual((s.respondents, s.anomalies), (7, 0))
        self.assertAlmostEqual(s.mean, 45 / 7)
        self.assertAlmostEqual(s.variance, 250 / 49)

    def test_boundary_exactness(self):
        # 恰在边界：|x-μ| == 2σ 必须不算异常（严格大于才异常）
        idx = GroupAnomalyIndex()
        for rid in "abcd":
            idx.add_rating(rid, 5)
        entered, exited = idx.add_rating("t", 10)
        self.assertEqual((entered, exited), (set(), set()))
        self.assertEqual(idx.summary().anomalies, 0)


class RandomStreamTest(unittest.TestCase):
    """小规模随机投稿流与参考实现逐步对拍。"""

    def _check_equal(self, idx, ref):
        n, anom, mean, var = ref.summary()
        s = idx.summary()
        self.assertEqual(s.respondents, n)
        self.assertEqual(s.anomalies, anom)
        self.assertAlmostEqual(s.mean, mean, places=9)
        self.assertAlmostEqual(s.variance, var, places=9)
        ref_anom = ref.anomaly_set()
        got_anom = idx.anomalies()
        self.assertEqual(set(got_anom), set(ref_anom))
        for rid, m in ref_anom.items():
            self.assertAlmostEqual(got_anom[rid], float(m), places=9)

    def test_random_stream(self):
        rng = random.Random(20261002)
        idx, ref = GroupAnomalyIndex(), RefModel()
        rids = [f"r{i}" for i in range(30)]
        # 投稿次数极不均衡：少数受访者权重极高
        weights = [1] * 24 + [40] * 4 + [300] * 2
        for _ in range(1200):
            rid = rng.choices(rids, weights=weights)[0]
            # 低分与高分同时异常：两端受访者偏态打分
            if rid in ("r0", "r1"):
                score = rng.choice([1, 1, 2])
            elif rid in ("r28", "r29"):
                score = rng.choice([9, 10, 10])
            else:
                score = rng.randint(3, 7)
            got = idx.add_rating(rid, score)
            exp = ref.add_rating(rid, score)
            self.assertEqual(got, exp, f"diff mismatch at {rid}={score}")
            self._check_equal(idx, ref)
        # 确认场景中确实同时出现过低端与高端异常
        anom = idx.anomalies()
        self.assertTrue(any(m < 3 for m in anom.values()))
        self.assertTrue(any(m > 8 for m in anom.values()))

    def test_repeated_ratings_same_respondent(self):
        # 同一受访者连续投稿，均分改变但仍异常时不应重复进入
        rng = random.Random(7)
        idx, ref = GroupAnomalyIndex(), RefModel()
        for i in range(20):
            idx.add_rating(f"m{i}", 5)
            ref.add_rating(f"m{i}", 5)
        for _ in range(300):
            rid = rng.choice(["out"] + [f"m{i}" for i in range(20)])
            score = 1 if rid == "out" else rng.randint(4, 6)
            self.assertEqual(idx.add_rating(rid, score), ref.add_rating(rid, score))
            self._check_equal(idx, ref)


class RoundingTest(unittest.TestCase):
    def test_half_up(self):
        self.assertEqual(_round6(3, 2), 1_500_000)          # 1.5
        self.assertEqual(_round6(10, 3), 3_333_333)         # 3.3333333…
        self.assertEqual(_round6(1, 6), 166_667)            # 0.1666666…
        self.assertEqual(_round6(129, 128), 1_007_813)      # 1.0078125 恰半，向上
        self.assertEqual(_round6(130, 128), 1_015_625)      # 1.015625 整
        self.assertEqual(_round6(131, 128), 1_023_438)      # 1.0234375 恰半，向上
        for total, count in [(129, 128), (10, 3), (1, 6), (7, 13)]:
            self.assertEqual(_round6(total, count), _ref_round6(total, count))

    def test_rounding_boundary_integration(self):
        # 受访者 128 条评分合计 129 -> 均分 1.0078125 -> 向上舍入 1.007813
        idx, ref = GroupAnomalyIndex(), RefModel()
        for i in range(30):
            idx.add_rating(f"n{i}", 5)
            ref.add_rating(f"n{i}", 5)
        for i in range(127):
            self.assertEqual(idx.add_rating("edge", 1), ref.add_rating("edge", 1))
        self.assertEqual(idx.add_rating("edge", 2), ref.add_rating("edge", 2))
        anom = idx.anomalies()
        self.assertIn("edge", anom)
        self.assertAlmostEqual(anom["edge"], 1.007813, places=9)
        self.assertEqual(set(anom), set(ref.anomaly_set()))


class EdgeCaseTest(unittest.TestCase):
    def test_empty(self):
        idx = GroupAnomalyIndex()
        self.assertEqual(idx.summary(), Summary(0, 0, 0.0, 0.0))
        self.assertEqual(idx.anomalies(), {})

    def test_zero_variance_no_anomaly(self):
        idx = GroupAnomalyIndex()
        for i in range(50):
            self.assertEqual(idx.add_rating(f"r{i}", 7), (set(), set()))
        self.assertEqual(idx.summary().anomalies, 0)
        self.assertEqual(idx.anomalies(), {})

    def test_invalid_score(self):
        idx = GroupAnomalyIndex()
        for bad in (0, 11, -1, 2.5, "5", True, None):
            with self.assertRaises(ValueError):
                idx.add_rating("r", bad)

    def test_summary_counts_consistent(self):
        rng = random.Random(99)
        idx = GroupAnomalyIndex()
        for _ in range(500):
            idx.add_rating(f"r{rng.randrange(15)}", rng.randint(1, 10))
        self.assertEqual(idx.summary().anomalies, len(idx.anomalies()))


if __name__ == "__main__":
    unittest.main()
