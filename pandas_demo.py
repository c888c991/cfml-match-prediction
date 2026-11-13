# -*- coding: utf-8 -*-
"""
CFML 赛事数据 · Pandas / NumPy / Matplotlib 实操验证

目的：用真实数据跑一遍「读取 → 清洗 → 分组统计 → 可视化」，
      产出可写进简历、也能在面试时当场复现的代码。

运行：python pandas_demo.py
输出：控制台统计 + analysis_charts.png
"""
import os
import sqlite3

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "cfml.db")
OUT = os.path.join(HERE, "analysis_charts.png")

# 中文显示
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

con = sqlite3.connect(DB)

# ============================================================
# 1. 读取：一次 JOIN 多表（Pandas read_sql 直接吃 SQL）
# ============================================================
print("=" * 62)
print("1. 读取数据")
print("=" * 62)

sql_map = """
SELECT m.matchid, m.match_time, c.name AS contest, m.teamA, m.teamB,
       mp.map_seq, mp.scoreA, mp.scoreB
FROM map mp
JOIN match m   ON m.matchid = mp.matchid
JOIN contest c ON c.contestid = mp.contestid
"""
df = pd.read_sql(sql_map, con)
print("地图级记录:", df.shape)

# match_time 存的是文本，转成 datetime 才能按赛季分组
df["match_time"] = pd.to_datetime(df["match_time"], errors="coerce")
print("时间转换后无效值:", df["match_time"].isna().sum())

# 从时间派生「赛季」字段（9 月开学制：9-12 -> 当年, 1-8 -> 上一年）
df["season"] = np.where(df["match_time"].dt.month >= 9,
                        df["match_time"].dt.year,
                        df["match_time"].dt.year - 1)


# ============================================================
# 2. 清洗：缺失 / 重复 / 异常
# ============================================================
print("\n" + "=" * 62)
print("2. 数据清洗")
print("=" * 62)

before = len(df)

# 2.1 缺失值检查
na = df.isna().sum()
print("缺失值（仅列有缺失）:")
print(na[na > 0].to_string() if (na > 0).any() else "  无")

# 2.2 去重
dup = df.duplicated(subset=["matchid", "map_seq"]).sum()
df = df.drop_duplicates(subset=["matchid", "map_seq"])
print("重复记录剔除:", dup)

# 2.3 比分合法性：CFML 单图最多打到十几回合，大于 25 视为脏数据
bad = df[(df["scoreA"] > 25) | (df["scoreB"] > 25) | (df["scoreA"] < 0)]
print("比分异常剔除:", len(bad))
df = df.drop(index=bad.index)

# 2.4 平局在 BO 赛制里不可能出现，单独标记（不删，供核对）
draw = (df["scoreA"] == df["scoreB"]).sum()
print("平局（应为 0）:", draw)

# 2.5 名称归一化：这是这个项目里最实的一步清洗
#     原始数据里队名带赞助商后缀 / 缩写 / 简写，不统一就没法按队伍聚合
name_map = (pd.read_sql("SELECT teamA_raw, teamA FROM map", con)
              .drop_duplicates()
              .rename(columns={"teamA_raw": "原始", "teamA": "标准"}))
n_raw, n_std = name_map["原始"].nunique(), name_map["标准"].nunique()
print("\n队名归一化：%d 个原始写法 -> %d 个标准队名" % (n_raw, n_std))

# 一对多的那些，就是最容易让统计出错的脏数据
multi = (name_map.groupby("标准")["原始"]
                 .agg(["nunique", lambda s: " / ".join(sorted(s))])
                 .rename(columns={"nunique": "写法数", "<lambda_0>": "原始写法"}))
multi = multi[multi["写法数"] > 1].sort_values("写法数", ascending=False)
if len(multi):
    print("需要合并的（一个队有多种写法）:")
    print(multi.to_string())

print("\n清洗后:", len(df), "条（原始 %d，剔除 %.1f%%）"
      % (before, (before - len(df)) / before * 100))


# ============================================================
# 3. 分组统计：groupby / agg / pivot
# ============================================================
print("\n" + "=" * 62)
print("3. 分组统计")
print("=" * 62)

# 3.1 每个赛季：赛程数、地图数、参赛队数
per_season = (df.groupby("season")
                .agg(地图数=("matchid", "size"),
                     比赛数=("matchid", "nunique"),
                     队伍数=("teamA", "nunique")))
print("\n【按赛季】")
print(per_season.to_string())

# 3.2 队伍胜率：把 A/B 两侧「融化」成长表再统一统计
#     这是真实项目里最常用的技巧——宽表转长表后 groupby 一次搞定
win = pd.concat([
    df.rename(columns={"teamA": "team", "scoreA": "s", "scoreB": "o"})[["team", "s", "o"]],
    df.rename(columns={"teamB": "team", "scoreB": "s", "scoreA": "o"})[["team", "s", "o"]],
])
win["won"] = (win["s"] > win["o"]).astype(int)

team_stat = (win.groupby("team")
                .agg(场次=("won", "size"), 胜场=("won", "sum")))
team_stat["胜率"] = (team_stat["胜场"] / team_stat["场次"]).round(3)
team_stat = team_stat[team_stat["场次"] >= 20].sort_values("胜率", ascending=False)

print("\n【队伍胜率 Top 10（场次 ≥ 20）】")
print(team_stat.head(10).to_string())

# 3.3 地图节奏：每张地图总回合数
df["总回合"] = df["scoreA"] + df["scoreB"]
print("\n【地图回合数分布】")
print(df["总回合"].describe().round(2).to_string())

# 3.4 用 NumPy 算两个统计量的相关系数（比分差 vs 总回合）
diff = np.abs(df["scoreA"] - df["scoreB"].values)
corr = np.corrcoef(diff, df["总回合"].values)[0, 1]
print("\n比分差 与 总回合 的皮尔逊相关系数: %.3f" % corr)


# ============================================================
# 4. 可视化：三联图
# ============================================================
print("\n" + "=" * 62)
print("4. 可视化 ->", os.path.basename(OUT))
print("=" * 62)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
fig.suptitle("CFML 赛事数据概览（Pandas / NumPy / Matplotlib）",
             fontsize=13, fontweight="bold")

# (1) 各赛季地图数
ax = axes[0]
per_season["地图数"].plot(kind="bar", ax=ax, color="#1b3a54", width=.55)
ax.set_title("各赛季地图数", fontsize=11)
ax.set_xlabel("赛季")
ax.set_ylabel("地图数")
ax.grid(axis="y", alpha=.25)
for i, v in enumerate(per_season["地图数"].values):
    ax.text(i, v + 8, str(v), ha="center", fontsize=9)

# (2) 队伍胜率 Top 10 横向条
ax = axes[1]
top = team_stat.head(10).iloc[::-1]
ax.barh(top.index, top["胜率"], color="#3d6b93", height=.62)
ax.set_title("队伍胜率 Top 10（≥20 场）", fontsize=11)
ax.set_xlim(0, 1)
ax.set_xlabel("胜率")
ax.grid(axis="x", alpha=.25)
for y, v in enumerate(top["胜率"].values):
    ax.text(v + .01, y, "%.0f%%" % (v * 100), va="center", fontsize=8.5)

# (3) 地图总回合数分布
ax = axes[2]
ax.hist(df["总回合"], bins=range(0, 26), color="#7fa4c4", edgecolor="white")
ax.axvline(df["总回合"].mean(), color="#c0392b", ls="--", lw=1.4,
           label="均值 %.1f" % df["总回合"].mean())
ax.set_title("单图总回合数分布", fontsize=11)
ax.set_xlabel("总回合")
ax.set_ylabel("地图数")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=.25)

plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print("已保存:", OUT)

con.close()
