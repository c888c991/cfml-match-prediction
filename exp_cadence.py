# -*- coding: utf-8 -*-
"""exp_cadence.py —— 实验：冷启动量与重训节奏该取多少？

WARMUP（前 N 场只训不评）和 RETRAIN（每 N 场重训一次）一直是拍的 80/50。
⚠️ 公平比法：**固定同一批评估样本**（eval_from=200），只变训练起点与重训间隔。
不固定的话，warmup 越大跳过的早期比赛越多，命中率虚高只是样本变容易了。
"""
import backtest as B

rows = B.load()

print("=" * 72)
print("冷启动量 / 重训节奏扫描（滚动前向，评估样本固定从第 200 场起）")
print("=" * 72)
print(f'{"训练起点":>10}{"重训间隔":>10}{"命中率":>10}{"对数损失":>11}')
print("-" * 72)
for warm, rt in ((0, 50), (80, 50), (160, 50), (200, 50),
                 (200, 25), (200, 100), (200, 200), (200, 400)):
    a, ll = B.run(rows, quiet=True, warmup=warm, retrain=rt, eval_from=200)
    tag = "开赛首场" if warm == 0 else f"第 {warm} 场"
    print(f'{tag:>10}{rt:>10}{a*100:>9.1f}%{ll:>11.4f}')
print("-" * 72)
print("对数损失越小越好。训练起点越小 = 早期数据越多；重训间隔越小 = 模型越新。")
print("=" * 72)
