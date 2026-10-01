"""规模用例：20 万受访者 + 100 万条追加评分。

- 5% 受访者为离群者（均分恒为 1 或 10），常驻异常 >= 1 万；
- 追加阶段多数更新只改变群体统计量，不改变异常名单；
- 在检查点用独立的全量重算核对完整异常集合与摘要；
- 打印耗时与翻转计数，验证未按次遍历全部异常者。

用法：python3 scale_test.py          # 完整规模
      python3 scale_test.py quick    # 缩小规模快速自检
"""
import random
import sys
import time
from math import isqrt

from anomaly_index import GroupAnomalyIndex, _round6

_SCALE = 1_000_000


def full_recompute(totals):
    """独立全量重算：从测试侧原始记录扫描出完整异常集合与摘要。"""
    means = {rid: _round6(t, c) for rid, (t, c) in totals.items()}
    n = len(means)
    if n == 0:
        return set(), 0.0, 0.0
    s = sum(means.values())
    q = sum(x * x for x in means.values())
    mean = s / (n * _SCALE)
    var = (q * n - s * s) / (n * n * _SCALE * _SCALE)
    t = 4 * (q * n - s * s)
    if t <= 0:
        return set(), mean, var
    r = isqrt(t)
    low_max = (s - r - 1) // n
    high_min = (s + r + n) // n
    anom = {rid for rid, x in means.items() if x <= low_max or x >= high_min}
    return anom, mean, var


def check(idx, totals, label):
    anom, mean, var = full_recompute(totals)
    s = idx.summary()
    assert s.respondents == len(totals), label
    assert s.anomalies == len(anom), f"{label}: {s.anomalies} != {len(anom)}"
    assert set(idx.anomalies()) == anom, label
    assert abs(s.mean - mean) < 1e-9 and abs(s.variance - var) < 1e-9, label
    print(f"[check] {label}: R={s.respondents} anomalies={s.anomalies} "
          f"mean={s.mean:.6f} var={s.variance:.6f} OK")
    return len(anom)


def main() -> None:
    quick = len(sys.argv) > 1 and sys.argv[1] == "quick"
    n_resp = 20_000 if quick else 200_000
    n_add = 100_000 if quick else 1_000_000
    rng = random.Random(7)
    idx = GroupAnomalyIndex()
    totals = {}

    def feed(rid, score):
        rec = totals.setdefault(rid, [0, 0])
        rec[0] += score
        rec[1] += 1
        e, x = idx.add_rating(rid, score)
        return len(e) + len(x)

    # 阶段一：建立受访者。每 20 人 1 个离群（一半 1 分一半 10 分）。
    t0 = time.perf_counter()
    for i in range(n_resp):
        if i % 20 == 0:
            score = 1 if (i // 20) % 2 == 0 else 10
        else:
            score = rng.choice((4, 5, 5, 6))
        feed(f"r{i}", score)
    t1 = time.perf_counter()
    print(f"build {n_resp} respondents: {t1 - t0:.2f}s")
    n_anom = check(idx, totals, "after build")
    assert n_anom >= n_resp // 20, n_anom

    # 阶段二：追加评分。离群者维持离群分，中间群体在 4..6 内随机，
    # 因此多数更新只改变群体统计量而不改变异常名单。
    flips = 0
    changed_stats = 0
    t2 = time.perf_counter()
    for j in range(n_add):
        i = rng.randrange(n_resp)
        if i % 20 == 0:
            score = 1 if (i // 20) % 2 == 0 else 10
        else:
            score = rng.choice((4, 5, 6))
        before = (idx._sum, idx._sq)
        flips += feed(f"r{i}", score)
        if (idx._sum, idx._sq) != before:
            changed_stats += 1
        if not quick and j == n_add // 2:
            check(idx, totals, "midway")
    t3 = time.perf_counter()
    print(f"append {n_add} ratings: {t3 - t2:.2f}s "
          f"({(t3 - t2) / n_add * 1e6:.1f}us/op)")
    n_anom = check(idx, totals, "after append")
    assert n_anom >= n_resp // 20, n_anom
    print(f"status flips: {flips} (of {n_add} ops); "
          f"updates changing group stats: {changed_stats}")
    assert changed_stats > n_add * 0.7
    assert flips < n_add * 0.01
    print("scale test OK")


if __name__ == "__main__":
    main()
