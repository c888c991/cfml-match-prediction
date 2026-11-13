# -*- coding: utf-8 -*-
"""exp_kalman.py —— 实验：把「手工半衰期」换成「卡尔曼滤波」跟踪队伍实力

## 现在是什么

局前预测里，「球队强弱」是用**时间衰减加权胜率**表示的：

    wr(队伍) = Σ w(age) · won  /  Σ w(age)     其中 w(age) = 0.5 ** (age / 365)

再取两队之差 `wr_diff = a_wr - b_wr` 当模型输入。半衰期 365 天是**手工拍**的，
而且对所有队伍、所有时期一视同仁 —— 这显然不真实：
刚换血的队，它的历史应该快速失效；阵容稳定的老队，历史可以多留一点。

## 卡尔曼能替什么

把「队伍实力」当成**隐状态**，每打一场就更新一次：

    状态:   x_i        （第 i 队的实力，logit 尺度）
    观测:   y ∈ {0,1}  （A 赢没赢）
    观测方程: P(A 赢) = sigmoid(x_A - x_B)
    过程噪声: 每周/每天给 P 加一点，代表"时间流逝让实力变得不确定"

它的好处是**每个队伍各自带一个不确定性 P**：
- 新队 / 刚换血的队 P 大 → 观测权重大 → 快速调整
- 老队 P 小 → 观测权重小 → 实力平滑

等于把"一个全局半衰期"换成"每队自适应的学习率"。

## ⚠️ 一个必须说清的技术前提

**这不是标准卡尔曼滤波。** 标准卡尔曼要求「线性观测 + 高斯噪声」，
而这里是二值观测、观测函数是 sigmoid（非线性）。
所以真正实现的是**扩展卡尔曼滤波（EKF）**：对 sigmoid 做一阶泰勒展开，
用雅可比 H 近似。Glicko 评分也是同一类东西（卡尔曼思路的近似）。

## 实验设计

把 FEATS 第 0 维（wr_diff）换成 EKF 算出的实力差 `kf_diff`，其余 14 维不变，
滚动前向回测（与 backtest.py 同口径：前 80 场冷启动、每 50 场重训一次），
只评估「开赛前」的预测点，对比命中率 / 对数损失 / 布里尔分数。

无泄漏：EKF 记录的是**每场比赛更新之前**的实力估计，所以只用了该场之前的信息。

用法：
    python exp_kalman.py                 # 扫几档过程噪声
    python exp_kalman.py --q 0.001       # 单档，顺便打印权重对照
"""
import argparse
import math
import sqlite3
from collections import defaultdict

from cfml_core import (FEATS, featurize, train_logreg, sigmoid_predict,
                       load_all, connect, parse_time)

DB = "cfml.db"
WARMUP = 80
RETRAIN = 50


# ---------------------------------------------------------------- 扩展卡尔曼滤波
def sigmoid(z):
    z = max(-30.0, min(30.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def run_ekf(ms, q=0.001, r=0.25, sigma0=1.0):
    """跑扩展卡尔曼滤波，返回 (pre, teams, x_final)。

    pre     = {matchid: (x_a, x_b)}，都是**开赛前**的估计（无泄漏）
    teams   = 队伍顺序
    x_final = 末状态实力（只为看排序，不参与评估）

    q      过程噪声率（每天给协方差加多少）：越大 = 老信息越快失效，等价于"半衰期越短"
    r      观测噪声（二值观测，0.25 是伯努利方差）
    sigma0 初始实力标准差（新队有多不确定）
    """
    teams = sorted({m["teamA"] for m in ms} | {m["teamB"] for m in ms})
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    x = [0.0] * n
    P = [[(sigma0 ** 2 if i == j else 0.0) for j in range(n)] for i in range(n)]

    pre = {}
    last = None
    for m in ms:
        t = parse_time(m["match_time"])
        # 时间流逝 -> 不确定性增长（这就是"遗忘"）
        if last is not None and t is not None:
            dt = (t - last).total_seconds() / 86400.0
            if dt > 0:
                add = q * dt
                for i in range(n):
                    P[i][i] += add
        if t is not None:
            last = t

        a, b = idx[m["teamA"]], idx[m["teamB"]]
        pre[m["matchid"]] = (x[a], x[b], P[a][a], P[b][b])   # 先记录，再更新

        w = m["winner"]
        if w not in (m["teamA"], m["teamB"]):
            continue                               # 结果不明，不更新

        d = x[a] - x[b]
        p = sigmoid(d)
        Ha = p * (1 - p)
        Hb = -Ha
        S = (Ha * Ha * P[a][a] + Ha * Hb * P[a][b]
             + Hb * Ha * P[b][a] + Hb * Hb * P[b][b] + r)
        if S <= 0:
            continue
        K = [(P[i][a] * Ha + P[i][b] * Hb) / S for i in range(n)]
        y = 1.0 if w == m["teamA"] else 0.0
        inn = y - p
        for i in range(n):
            x[i] += K[i] * inn
        HP = [Ha * P[a][j] + Hb * P[b][j] for j in range(n)]
        for i in range(n):
            ki = K[i]
            if ki == 0.0:
                continue
            Pi = P[i]
            for j in range(n):
                Pi[j] -= ki * HP[j]
    return pre, teams, x


# ---------------------------------------------------------------- 回测
def load_points():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    rows = [dict(r) for r in c.execute(
        "SELECT * FROM feature_point WHERE y_match IS NOT NULL AND match_time IS NOT NULL "
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


def metrics(res):
    n = len(res)
    acc = sum(1 for p, t in res if (p >= 0.5) == (t == 1)) / n
    ll = br = 0.0
    for p, t in res:
        p = min(max(p, 1e-6), 1 - 1e-6)
        ll += -(t * math.log(p) + (1 - t) * math.log(1 - p))
        br += (p - t) ** 2
    return acc, ll / n, br / n, n


def _ll(p, t):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return -(t * math.log(p) + (1 - t) * math.log(1 - p))


def pair_test(resA, resB):
    """配对比较逐样本对数损失（同一批预测点）。返回 (平均差, 标准误, t 值)

    平均差 < 0 表示 B 比 A 好。|t| > 2 才算统计上站得住。
    """
    if len(resA) != len(resB):
        return None
    d = [_ll(pa, ta) - _ll(pb, tb) for (pa, ta), (pb, tb) in zip(resA, resB)]
    n = len(d)
    mean = sum(d) / n
    var = sum((v - mean) ** 2 for v in d) / (n - 1)
    se = math.sqrt(var / n)
    return mean, se, (mean / se if se else 0.0)


def backtest(rows, kf=None, use_conf=False):
    """kf=None 用现状（wr_diff）；传了就把第 0 维换成卡尔曼实力差。

    use_conf=True 时，连第 4 维（exp_diff「经验量差」）也换成卡尔曼的不确定性差：
    协方差小 = 信息足 = 确定，等价于"经验多"。用 log 精度（-log P）来对齐原语义。
    """
    mids, by_match = group_by_match(rows)

    def feat(r):
        v = featurize(r)
        if kf is not None:
            pa, pb, va, vb = kf.get(r["matchid"], (0.0, 0.0, 1.0, 1.0))
            v = list(v)
            v[0] = pa - pb
            if use_conf:
                v[4] = math.log(max(vb, 1e-9)) - math.log(max(va, 1e-9))
        return v

    res, cache, model = [], [], None
    for i, mid in enumerate(mids):
        if i >= WARMUP:
            if (i - WARMUP) % RETRAIN == 0:
                X, y = [], []
                for prev in cache:
                    for r in by_match[prev]:
                        if r["y_match"] is not None:
                            X.append(feat(r))
                            y.append(r["y_match"])
                model = train_logreg(X, y)
            for r in by_match[mid]:
                if r["mapseq"] == 0 and r["y_match"] is not None:
                    res.append((sigmoid_predict(model, feat(r)), r["y_match"]))
        cache.append(mid)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=float, nargs="*", default=[0.0005, 0.001, 0.002, 0.004, 0.01])
    ap.add_argument("--r", type=float, default=0.25)
    ap.add_argument("--sigma0", type=float, default=1.0)
    args = ap.parse_args()

    rows = load_points()
    ms, _, _ = load_all(connect())

    res_base = backtest(rows)
    base = metrics(res_base)
    print("=" * 84)
    print("实验：手工半衰期（现状） vs 扩展卡尔曼滤波跟踪实力")
    print("=" * 84)
    print(f"评估 {base[3]} 个赛前预测点，滚动前向；观测噪声 r={args.r}，初始 sigma0={args.sigma0}\n")
    print(f'{"方案":<38}{"命中率":>8}{"对数损失":>11}{"布里尔":>10}')
    print("-" * 84)
    print(f'{"现状：衰减加权胜率差（半衰期 365 天）":<38}'
          f'{base[0]*100:>7.1f}%{base[1]:>11.4f}{base[2]:>10.4f}')

    best = None   # (metrics, res, tag, kf, teams, x_final, q, use_conf)
    for q in args.q:
        kf, teams, x_final = run_ekf(ms, q=q, r=args.r, sigma0=args.sigma0)
        for uc, sub in ((False, "只换实力维"), (True, "实力维+不确定性维")):
            res = backtest(rows, kf, use_conf=uc)
            m = metrics(res)
            print(f'{f"卡尔曼 q={q}：{sub}":<38}'
                  f'{m[0]*100:>7.1f}%{m[1]:>11.4f}{m[2]:>10.4f}')
            if best is None or m[1] < best[0][1]:
                best = (m, res, f"q={q} {sub}", kf, teams, x_final, q, uc)

    print("-" * 84)
    m, res_kf, tag, kf, teams, x_final, q, uc = best
    print(f"最好的一档：{tag}　命中率 {m[0]*100:.1f}%（现状 {base[0]*100:.1f}%，"
          f"差 {(m[0]-base[0])*100:+.1f}）　"
          f"对数损失 {m[1]:.4f}（现状 {base[1]:.4f}，差 {m[1]-base[1]:+.4f}）")

    pt = pair_test(res_base, res_kf)
    if pt:
        mean, se, t = pt
        verdict = ("显著（|t| > 2）" if abs(t) > 2 else "不显著（|t| <= 2），可能只是噪声")
        print(f"\n【配对显著性检验】逐样本对数损失，现状 − 上面这一档")
        print(f"  平均差 {mean:+.4f}　标准误 {se:.4f}　t 值 {t:+.2f}　→ {verdict}")

    print("\n【卡尔曼给出的队伍实力排序（末状态）】")
    print(f'{"队伍":<14}{"实力 x":>10}')
    for t, v in sorted(zip(teams, x_final), key=lambda z: -z[1]):
        print(f"{t:<14}{v:>10.3f}")
    print("=" * 84)


if __name__ == "__main__":
    main()
