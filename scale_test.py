"""规模用例：20 万受访者 + 追加 100 万条评分，常驻异常者 >= 1 万。

运行：python3 scale_test.py

校验方式：
- 影子异常集合完全由 add_rating 返回的差量增量维护；
- 每 5 万次投稿做一次 O(R) 暴力全量重算，与 anomalies()/summary()/影子集合三方比对；
- 统计并断言：常驻异常者 >= 1 万，且绝大多数投稿不改变异常名单。
"""

import random
import time

from anomaly_index import AnomalyIndex, round6

N_RESPONDENTS = 200_000
N_EXTRA_RATINGS = 1_000_000
CHECK_EVERY = 50_000
MIN_ANOMALIES = 10_000


def brute_force_anomalies(sums, cnts):
    """独立的 O(R) 全量重算：返回 (异常id集合, 群体均值, 总体方差)。"""
    keys = [round6(s, c) for s, c in zip(sums, cnts)]
    r = len(keys)
    s1 = sum(keys)
    s2 = sum(k * k for k in keys)
    n = s2 * r - s1 * s1
    anom = {i for i, k in enumerate(keys) if n > 0 and (k * r - s1) ** 2 > 4 * n}
    return anom, s1 / (r * 1e6), n / (r * r * 1e12)


def check(idx, sums, cnts, shadow, label):
    expected, mean, var = brute_force_anomalies(sums, cnts)
    got = idx.anomalies()
    assert set(got) == expected, f"{label}: anomalies() 与暴力重算不一致"
    assert shadow == expected, f"{label}: 差量累积的影子集合与暴力重算不一致"
    s = idx.summary()
    assert s.respondent_count == len(sums), f"{label}: 受访者数不一致"
    assert s.anomaly_count == len(expected), f"{label}: 异常人数不一致"
    assert abs(s.mean - mean) < 1e-9 and abs(s.variance - var) < 1e-9, f"{label}: 均值/方差不一致"
    return len(expected)


def main() -> None:
    rng = random.Random(20261002)
    # 3% 低分群(1-2)、3% 高分群(9-10)、其余中间群(4-6) => 双侧常驻异常
    groups = []
    for _ in range(N_RESPONDENTS):
        p = rng.random()
        groups.append(0 if p < 0.03 else (1 if p < 0.06 else 2))
    score_range = {0: (1, 2), 1: (9, 10), 2: (4, 6)}

    idx = AnomalyIndex()
    sums = [0] * N_RESPONDENTS
    cnts = [0] * N_RESPONDENTS
    shadow = set()
    total_flips = 0
    ops_with_flips = 0

    def feed(rid, score):
        nonlocal total_flips, ops_with_flips
        entered, exited = idx.add_rating(rid, score)
        sums[rid] += score
        cnts[rid] += 1
        shadow.update(entered)
        shadow.difference_update(exited)
        d = len(entered) + len(exited)
        total_flips += d
        ops_with_flips += d > 0

    t0 = time.perf_counter()
    for rid in range(N_RESPONDENTS):
        lo, hi = score_range[groups[rid]]
        feed(rid, rng.randint(lo, hi))
    t1 = time.perf_counter()
    n_anom = check(idx, sums, cnts, shadow, "建群后")
    print(f"建群 {N_RESPONDENTS} 人: {t1 - t0:.2f}s, 异常 {n_anom} 人")
    assert n_anom >= MIN_ANOMALIES, "常驻异常者不足 1 万"

    t2 = time.perf_counter()
    mids = [i for i, g in enumerate(groups) if g == 2]
    converted = []  # 被人为推入异常的中间群受访者，稍后拉回以触发解除
    for i in range(1, N_EXTRA_RATINGS + 1):
        rid = rng.randrange(N_RESPONDENTS)
        lo, hi = score_range[groups[rid]]
        feed(rid, rng.randint(lo, hi))
        if i % 100_000 == 50_000:
            victim = mids[i // 100_000]
            for _ in range(10):
                feed(victim, 10)  # 均分推高越过阈值 => 进入异常
            converted.append(victim)
        if i % 100_000 == 0 and converted:
            victim = converted.pop(0)
            for _ in range(10):
                feed(victim, 1)  # 均分拉回 => 解除异常
        if i % CHECK_EVERY == 0:
            n_anom = check(idx, sums, cnts, shadow, f"第 {i} 条")
            assert n_anom >= MIN_ANOMALIES, f"第 {i} 条后常驻异常者不足 1 万: {n_anom}"
    t3 = time.perf_counter()

    n_anom = check(idx, sums, cnts, shadow, "最终")
    total_ops = N_RESPONDENTS + N_EXTRA_RATINGS
    print(f"追加 {N_EXTRA_RATINGS} 条: {t3 - t2:.2f}s, 总耗时 {t3 - t0:.2f}s")
    print(f"总投稿 {total_ops} 条, 平均 {(t3 - t0) / total_ops * 1e6:.1f}µs/条")
    print(f"最终异常 {n_anom} 人; 翻转总人次 {total_flips}, "
          f"有翻转的投稿 {ops_with_flips} 次 ({ops_with_flips / total_ops:.2%})")
    assert n_anom >= MIN_ANOMALIES
    assert ops_with_flips < total_ops * 0.5, "多数投稿不应改变异常名单"
    print("规模用例通过")


if __name__ == "__main__":
    main()
