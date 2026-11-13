# -*- coding: utf-8 -*-
"""exp_nn.py —— 实验：局前预测该不该上神经网络？

## 背景

现在的模型是 15 维特征 + 逻辑回归（手写梯度下降、Z-score 标准化、L2=0.002）。
红线③写的是「不要换模型」，理由是：2304 个预测点落在 12 赛季 × 11 队 = 132 个单元里，
有效自由度远小于样本数，上复杂模型只会过拟合。

但这是推理，不是实测。这个脚本就用同一批数据、同一条滚动前向，
把「逻辑回归」和「小型神经网络」摆在一起比，用数据说话。

## 实现说明

环境里没有 numpy / sklearn（项目要求零第三方依赖），所以 MLP 是**纯 Python 手写**：
    15 维 -> H 个 tanh 隐单元 -> 1 维 sigmoid
Adam 优化器、L2 正则、全批量梯度下降。

对比口径与 backtest.py 完全一致（前 80 场冷启动、滚动前向、只评估开赛前的点）。

用法：
    python exp_nn.py                # 默认 H=8 / H=16 各跑一遍
    python exp_nn.py --h 8 16 32    # 指定隐单元数
    python exp_nn.py --fast         # 少跑几轮，先看个大概
"""
import argparse
import math
import random
import sqlite3
import time
from collections import defaultdict

from cfml_core import featurize, train_logreg, sigmoid_predict

DB = "cfml.db"
WARMUP = 80
RETRAIN = 50


def sigmoid(z):
    z = max(-30.0, min(30.0, z))
    return 1.0 / (1.0 + math.exp(-z))


# ---------------------------------------------------------------- 手写 MLP
class MLP:
    """15 -> H -> 1，tanh 隐层 + sigmoid 输出，Adam 优化"""

    def __init__(self, d, h, seed=20260929):
        rnd = random.Random(seed)
        s1 = math.sqrt(6.0 / (d + h))          # Glorot 初始化
        s2 = math.sqrt(6.0 / (h + 1))
        self.h = h
        self.W1 = [[rnd.uniform(-s1, s1) for _ in range(d)] for _ in range(h)]
        self.b1 = [0.0] * h
        self.W2 = [rnd.uniform(-s2, s2) for _ in range(h)]
        self.b2 = 0.0
        # Adam 状态
        self.mW1 = [[0.0] * d for _ in range(h)]
        self.vW1 = [[0.0] * d for _ in range(h)]
        self.mb1 = [0.0] * h
        self.vb1 = [0.0] * h
        self.mW2 = [0.0] * h
        self.vW2 = [0.0] * h
        self.mb2 = 0.0
        self.vb2 = 0.0
        self.t = 0

    def _fwd(self, x):
        z1 = []
        for row, b in zip(self.W1, self.b1):
            s = 0.0
            for w, xv in zip(row, x):
                s += w * xv
            z1.append(s + b)
        hh = [math.tanh(v) for v in z1]
        z2 = self.b2
        for w, hv in zip(self.W2, hh):
            z2 += w * hv
        return hh, sigmoid(z2)

    def predict(self, x):
        return self._fwd(x)[1]

    def fit(self, X, y, epochs=300, lr=0.02, l2=0.002):
        n = len(X)
        d = len(X[0])
        b1, b2 = 0.9, 0.999
        eps = 1e-8
        for _ in range(epochs):
            gW1 = [[0.0] * d for _ in range(self.h)]
            gb1 = [0.0] * self.h
            gW2 = [0.0] * self.h
            gb2 = 0.0
            for i in range(n):
                x = X[i]
                hh, p = self._fwd(x)
                e = p - y[i]
                gb2 += e
                for k in range(self.h):
                    gW2[k] += e * hh[k]
                    dh = e * self.W2[k] * (1.0 - hh[k] * hh[k])   # tanh'
                    if dh:
                        gb1[k] += dh
                        gk = gW1[k]
                        for j in range(d):
                            gk[j] += dh * x[j]
            self.t += 1
            c1 = 1.0 - b1 ** self.t
            c2 = 1.0 - b2 ** self.t
            inv = lr / n
            # 输出层
            for k in range(self.h):
                g = gW2[k] / n + l2 * self.W2[k]
                self.mW2[k] = b1 * self.mW2[k] + (1 - b1) * g
                self.vW2[k] = b2 * self.vW2[k] + (1 - b2) * g * g
                self.W2[k] -= inv * (self.mW2[k] / c1) / (math.sqrt(self.vW2[k] / c2) + eps)
            g = gb2 / n
            self.mb2 = b1 * self.mb2 + (1 - b1) * g
            self.vb2 = b2 * self.vb2 + (1 - b2) * g * g
            self.b2 -= lr * (self.mb2 / c1) / (math.sqrt(self.vb2 / c2) + eps)
            # 隐层
            for k in range(self.h):
                g = gb1[k] / n
                self.mb1[k] = b1 * self.mb1[k] + (1 - b1) * g
                self.vb1[k] = b2 * self.vb1[k] + (1 - b2) * g * g
                self.b1[k] -= lr * (self.mb1[k] / c1) / (math.sqrt(self.vb1[k] / c2) + eps)
                Wk, gk = self.W1[k], gW1[k]
                m, v = self.mW1[k], self.vW1[k]
                for j in range(d):
                    gj = gk[j] / n + l2 * Wk[j]
                    m[j] = b1 * m[j] + (1 - b1) * gj
                    v[j] = b2 * v[j] + (1 - b2) * gj * gj
                    Wk[j] -= inv * (m[j] / c1) / (math.sqrt(v[j] / c2) + eps)


def standardize(X):
    n, d = len(X), len(X[0])
    mu = [sum(X[i][j] for i in range(n)) / n for j in range(d)]
    sd = []
    for j in range(d):
        v = sum((X[i][j] - mu[j]) ** 2 for i in range(n)) / n
        sd.append(math.sqrt(v) or 1.0)
    return [[(X[i][j] - mu[j]) / sd[j] for j in range(d)] for i in range(n)], mu, sd


# ---------------------------------------------------------------- 数据 / 评估
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
    d = [_ll(pa, ta) - _ll(pb, tb) for (pa, ta), (pb, tb) in zip(resA, resB)]
    n = len(d)
    mean = sum(d) / n
    var = sum((v - mean) ** 2 for v in d) / (n - 1)
    se = math.sqrt(var / n)
    return mean, se, (mean / se if se else 0.0)


def run(rows, kind, h=8, epochs=300, seed=20260929):
    """滚动前向。kind='lr' 用逻辑回归，kind='mlp' 用手写网络。"""
    mids, by_match = group_by_match(rows)
    res, cache, model, mu, sd = [], [], None, None, None
    for i, mid in enumerate(mids):
        if i >= WARMUP:
            if (i - WARMUP) % RETRAIN == 0:
                X, y = [], []
                for prev in cache:
                    for r in by_match[prev]:
                        if r["y_match"] is not None:
                            X.append(featurize(r))
                            y.append(r["y_match"])
                if kind == "lr":
                    model = train_logreg(X, y)
                else:
                    Z, mu, sd = standardize(X)
                    model = MLP(len(Z[0]), h, seed=seed)
                    model.fit(Z, y, epochs=epochs)
            for r in by_match[mid]:
                if r["mapseq"] == 0 and r["y_match"] is not None:
                    x = featurize(r)
                    if kind == "lr":
                        p = sigmoid_predict(model, x)
                    else:
                        p = model.predict([(x[j] - mu[j]) / sd[j] for j in range(len(x))])
                    res.append((p, r["y_match"]))
        cache.append(mid)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=int, nargs="*", default=[8, 16])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--fast", action="store_true", help="少数几轮，先看趋势")
    args = ap.parse_args()
    if args.fast:
        args.epochs = 60

    rows = load_points()
    print("=" * 80)
    print("实验：逻辑回归 vs 小型神经网络（同特征、同滚动前向、只评估开赛前）")
    print("=" * 80)

    t0 = time.time()
    res_lr = run(rows, "lr")
    m = metrics(res_lr)
    print(f'{"模型":<30}{"命中率":>9}{"对数损失":>12}{"布里尔":>11}')
    print("-" * 80)
    print(f'{f"逻辑回归（现状，15 特征）":<30}{m[0]*100:>8.1f}%{m[1]:>12.4f}{m[2]:>11.4f}'
          f'   用时 {time.time()-t0:.0f}s')

    for h in args.h:
        t0 = time.time()
        res_nn = run(rows, "mlp", h=h, epochs=args.epochs)
        mn = metrics(res_nn)
        mean, se, t = pair_test(res_lr, res_nn)
        verdict = "显著" if abs(t) > 2 else "不显著"
        print(f'{f"神经网络 15-{h}-1（{args.epochs} 轮）":<30}'
              f'{mn[0]*100:>8.1f}%{mn[1]:>12.4f}{mn[2]:>11.4f}'
              f'   用时 {time.time()-t0:.0f}s')
        print(f'{"   └ 相对逻辑回归（配对检验）":<30}'
              f'{(mn[0]-m[0])*100:>+8.1f}%{mn[1]-m[1]:>+12.4f}{mn[2]-m[2]:>+11.4f}'
              f'   t={t:+.2f}（{verdict}）')
    print("-" * 80)
    print(f"评估样本 {m[3]} 个赛前点，训练集含全部预测点（与现状口径一致）")
    print("=" * 80)
