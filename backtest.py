# -*- coding: utf-8 -*-
"""backtest.py —— 滚动前向回测（防数据泄漏）

## 为什么不能用随机划分

常见错误是 `train_test_split(shuffle=True)`。放在这里会双重泄漏：

1. **同场泄漏**：一场比赛有多个预测点（开赛前 / 打完 1 局 / 打完 2 局），标签是同一个比赛结果。
   随机划分会让"第 1 局后"落在训练集、"第 2 局后"落在测试集 —— 模型等于偷看了答案。
2. **跨时泄漏**：用 2026 年的比赛训练、去预测 2022 年的比赛。而这几年联赛的强弱格局、
   阵容、版本全变了（实测过：同一队名 4 年后主力 0/5 相同）。分数会虚高一大截，实盘立刻打回原形。

## 正确做法：滚动前向验证

按时间排序，预测第 i 场时**只用第 1..i-1 场**训练。逐场往前滚。
这是唯一能反映"实盘时你到底能拿到什么"的评估方式。

用法：
    python backtest.py                    # 默认半衰期 365 天
    python backtest.py --halflife 180     # 换半衰期再跑一遍
    python backtest.py --sweep            # 把几档半衰期都跑一遍做对比
"""
import argparse
import math
import os
import sqlite3
import sys
from collections import defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(BASE, "cfml.db")

WARMUP = 80      # 前 N 场比赛只用来积累训练数据，不计入评估
RETRAIN = 50     # 每 N 场重新训练一次模型

# 特征工程 / 逻辑回归 / 预测 全部统一走内核，别在这里另写一套。
# 内核改了这里会自动跟着变，两边口径永远一致。
from cfml_core import FEATS, featurize, train_logreg, sigmoid_predict as predict  # noqa: E402


# ------------------------------------------------------------------ 装载
def load():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    rows = c.execute("""
        SELECT * FROM feature_point
        WHERE y_match IS NOT NULL AND match_time IS NOT NULL
        ORDER BY match_time, matchid, mapseq
    """).fetchall()
    c.close()
    return [dict(r) for r in rows]


def group_by_match(rows):
    g = defaultdict(list)
    order = []
    for r in rows:
        if r["matchid"] not in g:
            order.append((r["match_time"], r["matchid"]))
        g[r["matchid"]].append(r)
    order.sort()
    return [mid for _, mid in order], g


# ------------------------------------------------------------------ 回测
def run(rows, quiet=False, warmup=None, retrain=None, eval_from=None):
    """warmup/retrain/eval_from 不传就用模块常量。

    ⚠️ 对比不同 warmup 时必须传同一个 eval_from，否则评估样本不同、数字不可比
    （warmup 越大跳过的早期比赛越多，命中率虚高只是样本变"容易"了）。
    """
    warmup = WARMUP if warmup is None else warmup
    retrain = RETRAIN if retrain is None else retrain
    eval_from = warmup if eval_from is None else eval_from
    mids, by_match = group_by_match(rows)

    # 每个预测点的两个基线
    #   基线 1：无脑猜赛程里排前面的那方（A）
    #   基线 2：猜"本赛季胜率更高"的一方
    results = []          # (mapseq, y_true, p_model)
    base_a = []           # 基线 1：无脑猜赛程里排前面的一方
    base_szn = []         # 基线 2：按本赛季胜率归一化出的概率

    model = None
    mid_cache = []
    j1 = []               # 打完 1 局后的点，用来诊断「模型有没有超过猜第 1 局胜者」

    for i, mid in enumerate(mids):
        pts = by_match[mid]
        if i >= warmup:
            if (i - warmup) % retrain == 0:
                X, y = [], []
                for prev in mid_cache:
                    for r in by_match[prev]:
                        if r["y_match"] is not None:
                            X.append(featurize(r))
                            y.append(r["y_match"])
                model = train_logreg(X, y)
            for r in pts:
                if r["y_match"] is None or i < eval_from:
                    continue
                p = predict(model, featurize(r))
                results.append((r["mapseq"], r["y_match"], p))
                if r["mapseq"] == 1:
                    j1.append((r["y_match"], p, r["score_diff"] or 0))
                # 基线
                base_a.append((r["mapseq"], r["y_match"], 0.5))
                sa, sb = r["a_sznwr"], r["b_sznwr"]
                # 软基线：把双方本赛季胜率归一化成概率。
                # 不能用硬 0/1 —— 一旦猜错对数损失会爆到 20 以上，整列数字失去意义。
                p_base = 0.5 if (sa is None or sb is None) else (sa + 0.5) / (sa + sb + 1.0)
                base_szn.append((r["mapseq"], r["y_match"], p_base))
        mid_cache.append(mid)

    if not results:
        print("样本不足，无法回测")
        return None

    def acc(rows):
        return sum(1 for _, t, p in rows if (p >= 0.5) == (t == 1)) / len(rows)

    def logloss(rows):
        s = 0.0
        for _, t, p in rows:
            p = min(max(p, 1e-6), 1 - 1e-6)
            s += -(t * math.log(p) + (1 - t) * math.log(1 - p))
        return s / len(rows)

    if quiet:
        return acc(results), logloss(results)

    print("=" * 78)
    print("滚动前向回测结果")
    print("=" * 78)
    print(f"评估样本 {len(results)} 个预测点，来自 {len(mids) - eval_from} 场比赛")
    print(f"（前 {warmup} 场只用于冷启动，不计入评估；每 {retrain} 场重训一次）\n")

    print(f'{"":<24}{"命中率":>9}{"对数损失":>11}')
    print(f'{"基线1：无脑猜 A":<24}{acc(base_a)*100:>8.1f}%{logloss(base_a):>11.4f}')
    print(f'{"基线2：猜本赛季胜率高方":<24}{acc(base_szn)*100:>8.1f}%{logloss(base_szn):>11.4f}')
    print(f'{"模型（逻辑回归）":<24}{acc(results)*100:>8.1f}%{logloss(results):>11.4f}')

    print(f'\n【按「什么时候预测」拆开看】—— 这才是重点')
    print(f'{"预测时机":<18}{"样本":>7}{"基线":>9}{"模型":>9}{"提升":>9}')
    for j in sorted(set(r[0] for r in results)):
        rs = [x for x in results if x[0] == j]
        bs = [x for x in base_szn if x[0] == j]
        if not rs:
            continue
        tag = "开赛前" if j == 0 else f"打完 {j} 局后"
        a_m, a_b = acc(rs) * 100, acc(bs) * 100
        print(f'{tag:<18}{len(rs):>7}{a_b:>8.1f}%{a_m:>8.1f}%{a_m - a_b:>+8.1f}')

    print(f'\n【对照：猜「第 1 局的胜者会赢下整场」】')
    one = [x for x in results if x[0] == 1]
    if one:
        print(f'  样本 {len(one)}   命中率 {acc(one)*100:.1f}%'
              f'   ← 这是实时场景下最傻但很强的对手，模型必须打赢它')

    if j1:
        agree = [(1, t, p) for t, p, sd in j1 if (p >= 0.5) == (sd > 0)]
        disagree = [(1, t, p) for t, p, sd in j1 if (p >= 0.5) != (sd > 0)]
        print(f'\n【诊断：模型在「打完 1 局后」到底有没有本事】')
        print(f'  跟「猜第 1 局胜者」同向的点：{len(agree)} 个'
              + (f'，命中率 {acc(agree)*100:.1f}%' if agree else ''))
        if disagree:
            print(f'  模型「反着猜」的点：{len(disagree)} 个，命中率 {acc(disagree)*100:.1f}%'
                  f'   ← 明显超过 50% 才说明模型有独立判断力')
        else:
            print(f'  模型从未反着猜 —— 它只是把「第 1 局胜者」复述了一遍，没有增量')

    # 只在"开赛前"那个点上评估，等价于按场计
    pre = [x for x in results if x[0] == 0]
    if pre:
        print(f'\n【按场计（只看每个预测点的开赛前）】')
        print(f'  样本 {len(pre)} 场   命中率 {acc(pre)*100:.1f}%')

    # 概率校准：分组看预测概率 vs 实际频率
    print(f'\n【概率校准】模型说 X% 的时候，实际发生了多少')
    bins = [(0, .4), (.4, .5), (.5, .6), (.6, .7), (.7, 1.01)]
    print(f'{"预测概率区间":<16}{"样本":>7}{"实际胜率":>10}')
    for lo, hi in bins:
        sub = [t for _, t, p in results if lo <= p < hi]
        if not sub:
            continue
        print(f'{f"{lo:.0%} ~ {hi:.0%}":<16}{len(sub):>7}{sum(sub)/len(sub)*100:>9.1f}%')

    return acc(results)


def sweep(rows):
    print("=" * 74)
    print("半衰期扫描（时间衰减到底取多少天）")
    print("=" * 74)
    print(f'{"半衰期":<16}{"命中率":>10}')
    import subprocess
    for hl in (90, 180, 365, 730, 0):
        subprocess.run([sys.executable, os.path.join(BASE, "build_features.py"),
                        "--halflife", str(hl)], capture_output=True)
        r = load()
        a, _ = run(r, quiet=True)
        tag = "不衰减" if hl == 0 else f"{hl} 天"
        print(f'{tag:<16}{a*100:>9.1f}%')
    # 还原默认
    subprocess.run([sys.executable, os.path.join(BASE, "build_features.py"),
                    "--halflife", "365"], capture_output=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--halflife", type=float, help="先按这个半衰期重建特征表再回测")
    ap.add_argument("--sweep", action="store_true", help="扫描多档半衰期")
    args = ap.parse_args()

    if args.halflife is not None:
        import subprocess
        subprocess.run([sys.executable, os.path.join(BASE, "build_features.py"),
                        "--halflife", str(args.halflife)], capture_output=True)

    data = load()
    if args.sweep:
        sweep(data)
    else:
        run(data)
