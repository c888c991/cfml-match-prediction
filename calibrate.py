# -*- coding: utf-8 -*-
"""calibrate.py —— 概率校准体检：模型说 X% 的时候，实际到底发生了多少

为什么单独做这个：
    回测的「命中率 72%」只说明排序对不对（谁更可能赢）。
    但要说「LGD 只有 11% 胜率」，靠的是概率本身的**刻度准不准**。
    逻辑回归很容易在两端过度自信 —— 系统性地把 80% 说成 92%、把 20% 说成 8%。
    如果不校准，看板上那些 11% / 88.9% 就只能当排序用，不能当概率读。

做法：
    跟 backtest.py 完全一样的滚动前向（防泄漏），但只收集「开赛前」的点，
    按预测概率细分成 12 档，逐档对比「平均预测」和「实际胜率」。

用法：
    python calibrate.py            # 赛前（mapseq=0）
    python calibrate.py --mapseq 1 # 看完 1 局后的点
"""
import argparse
import sqlite3
from collections import defaultdict

from cfml_core import featurize, train_logreg, sigmoid_predict as predict

DB = "cfml.db"
WARMUP = 80       # 跟 backtest.py 保持一致，两边数字才能对比
RETRAIN = 50


def load():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    rows = [dict(r) for r in c.execute(
        "SELECT * FROM feature_point "
        "WHERE y_match IS NOT NULL AND match_time IS NOT NULL "
        "ORDER BY match_time, matchid, mapseq")]
    c.close()
    return rows


def group_by_match(rows):
    g, order = defaultdict(list), []
    for r in rows:
        if r["matchid"] not in g:
            order.append((r["match_time"], r["matchid"]))
        g[r["matchid"]].append(r)
    order.sort()
    return [mid for _, mid in order], g


def out_of_sample(rows, mapseq):
    """滚动前向，收集 out-of-sample 的 (预测概率, 实际结果)"""
    mids, by_match = group_by_match(rows)
    got, model, cache = [], None, []
    for i, mid in enumerate(mids):
        if i >= WARMUP:
            if (i - WARMUP) % RETRAIN == 0:
                X, y = [], []
                for prev in cache:
                    for r in by_match[prev]:
                        if r["y_match"] is not None:
                            X.append(featurize(r))
                            y.append(r["y_match"])
                model = train_logreg(X, y)
            for r in by_match[mid]:
                if r["y_match"] is None or r["mapseq"] != mapseq:
                    continue
                got.append((predict(model, featurize(r)), r["y_match"]))
        cache.append(mid)
    return got


def report(got, title):
    n = len(got)
    if not n:
        print("没有样本")
        return
    print("=" * 78)
    print(f"概率校准体检 —— {title}")
    print("=" * 78)
    print(f"样本 {n} 个（滚动前向，out-of-sample）\n")

    # 先看整体：平均预测 vs 实际
    avg_p = sum(p for p, _ in got) / n
    act = sum(y for _, y in got) / n
    print(f"整体：平均预测 {avg_p*100:.1f}%　实际胜率 {act*100:.1f}%　"
          f"偏差 {(avg_p-act)*100:+.1f} 个百分点\n")

    # 细分箱：0~10%, 10~20%, ..., 90~100%
    print(f'{"预测区间":<12}{"样本":>6}{"占比":>8}{"平均预测":>10}{"实际胜率":>10}{"偏差":>9}')
    print("-" * 78)
    total_abs = 0.0
    for k in range(10):
        lo, hi = k / 10, (k + 1) / 10
        # 最后一档把 p==1.0 也收进来，别漏
        sub = ([(p, t) for p, t in got if p >= 0.9] if k == 9
               else [(p, t) for p, t in got if lo <= p < hi])
        if not sub:
            continue
        mp = sum(p for p, _ in sub) / len(sub)
        a = sum(t for _, t in sub) / len(sub)
        total_abs += abs(mp - a) * len(sub)
        print(f'{f"{lo:.0%}~{hi:.0%}":<12}{len(sub):>6}{len(sub)/n*100:>7.1f}%'
              f'{mp*100:>9.1f}%{a*100:>9.1f}%{(mp-a)*100:>+8.1f}')
    print("-" * 78)
    print(f"加权平均绝对偏差：{total_abs/n*100:.2f} 个百分点（越小越准）")

    # 极端区间单独点名 —— 看板上最惹眼的数字都在这里
    print("\n【极端区间】这些是看板上最惹眼的数字，单独看")
    for tag, sel in (("预测 < 20%（模型说很悬）", lambda p: p < 0.20),
                     ("预测 > 80%（模型说稳了）", lambda p: p > 0.80)):
        sub = [(p, t) for p, t in got if sel(p)]
        if not sub:
            print(f"  {tag}：没有样本")
            continue
        mp = sum(p for p, _ in sub) / len(sub)
        a = sum(t for _, t in sub) / len(sub)
        # 用「模型最自信的那批」判断可信度
        print(f"  {tag}：{len(sub)} 个　平均预测 {mp*100:.1f}%　"
              f"实际胜率 {a*100:.1f}%　→　{'高估了' if mp > a else '低估了'}"
              f" {abs(mp-a)*100:.1f} 个百分点")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapseq", type=int, default=0, help="0=赛前（默认），1=打完1局后")
    args = ap.parse_args()
    tag = "开赛前" if args.mapseq == 0 else f"打完 {args.mapseq} 局后"
    report(out_of_sample(load(), args.mapseq), tag)
