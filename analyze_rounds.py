# -*- coding: utf-8 -*-
"""analyze_rounds.py —— 「单局回合数」这条盘口到底能不能预测

已确认的单局格式（从逐回合流水 + 数据库比分反推）：
  · 先到 10 回合获胜（10:0 ~ 10:8）
  · 9:9 打平进加时，加时领先 2 分才结束（11:9 / 12:10 / 13:11 / 14:12 / 15:13）
  · 所以单局总回合数只可能是 10~18 以及 20/22/24/26/28（奇数 >18 不会出现）

这份脚本回答：
  1. 「回合数 > 16.5」的历史占比 = 押大/押小的裸基线
  2. 这个占比随时间（赛季）怎么变 —— 有没有版本/版本节奏的影响
  3. 它跟「两队实力差」有关吗 —— 越势均力敌越容易打长？
  4. 顺带把「第 1 局胜者赢整场」这条规则的真实命中率算准（局中预测的参考线）
"""
import os
import sqlite3
from collections import defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
c = sqlite3.connect(os.path.join(BASE, "cfml.db"))
c.row_factory = sqlite3.Row

# 1) 每局的回合数 + 所属赛季 + 该场开赛前的双方胜率
rows = c.execute("""
    SELECT m.matchid, m.map_seq, m.scoreA, m.scoreB, m.winner, m.teamA, m.teamB,
           mt.contestname, mt.match_time, mt.n_maps, mt.scoreA AS mAS, mt.scoreB AS mBS,
           fp.a_wr, fp.b_wr, fp.a_n, fp.b_n, fp.h2h_wr
    FROM map m
    JOIN match mt ON mt.matchid = m.matchid
    LEFT JOIN feature_point fp ON fp.matchid = m.matchid AND fp.mapseq = 0
    WHERE m.scoreA IS NOT NULL AND m.scoreB IS NOT NULL
      AND mt.is_official=1 AND mt.is_placeholder=0
    ORDER BY mt.match_time, m.matchid, m.map_seq
""").fetchall()

print(f"参与统计的单局：{len(rows)}")

def bucket(rd):
    return ">16.5" if rd > 16.5 else "<=16.5"

# ---------------------------------------------------------------- 1 + 2
print()
print("=" * 74)
print("1) 裸基线：无脑押「小（<=16.5）」能命中多少")
print("=" * 74)
over = sum(1 for r in rows if (r["scoreA"] + r["scoreB"]) > 16.5)
n = len(rows)
print(f"  大于 16.5 : {over:>5} 局  {over/n*100:5.1f}%")
print(f"  小于等于 16.5 : {n-over:>5} 局  {(n-over)/n*100:5.1f}%")
print(f"  → 无脑押『小』命中率 {(n-over)/n*100:.1f}%，几乎五五开，难怪庄家把线卡在 16.5")

print()
print("=" * 74)
print("2) 按赛季看：这条线的偏向稳不稳")
print("=" * 74)
by_season = defaultdict(lambda: [0, 0])
for r in rows:
    bs = by_season[r["contestname"]]
    bs[0] += 1
    if (r["scoreA"] + r["scoreB"]) > 16.5:
        bs[1] += 1
for s in sorted(by_season, key=lambda k: min(
        (x["match_time"] for x in rows if x["contestname"] == k), default="")):
    tot, ov = by_season[s]
    print(f"  {s:<22} {tot:>4} 局   大于16.5 占 {ov/tot*100:5.1f}%")

# ---------------------------------------------------------------- 3
print()
print("=" * 74)
print("3) 跟「两队实力差」有没有关系（势均力敌是不是更容易打长）")
print("=" * 74)

def diff_bucket(r):
    if r["a_wr"] is None or r["b_wr"] is None:
        return None
    d = abs(r["a_wr"] - r["b_wr"])
    if d < 0.10:
        return "① 极接近 (差<0.10)"
    if d < 0.20:
        return "② 接近 (0.10~0.20)"
    if d < 0.35:
        return "③ 有差距 (0.20~0.35)"
    return "④ 差距大 (>=0.35)"

buckets = defaultdict(lambda: [0, 0])
for r in rows:
    bk = diff_bucket(r)
    if bk is None:
        continue
    buckets[bk][0] += 1
    if (r["scoreA"] + r["scoreB"]) > 16.5:
        buckets[bk][1] += 1
print("  （a_wr/b_wr = 开赛前的衰减加权胜率）")
for bk in sorted(buckets):
    tot, ov = buckets[bk]
    print(f"  {bk:<24} {tot:>4} 局   大于16.5 占 {ov/tot*100:5.1f}%")

# 换个角度：胜率差本身跟回合数的相关
pairs = [(abs(r["a_wr"] - r["b_wr"]), r["scoreA"] + r["scoreB"])
         for r in rows if r["a_wr"] is not None and r["b_wr"] is not None]
if len(pairs) > 100:
    mx = sum(p[0] for p in pairs) / len(pairs)
    my = sum(p[1] for p in pairs) / len(pairs)
    cov = sum((x - mx) * (y - my) for x, y in pairs)
    sx = sum((x - mx) ** 2 for x, _ in pairs) ** 0.5
    sy = sum((y - my) ** 2 for _, y in pairs) ** 0.5
    rho = cov / (sx * sy) if sx and sy else 0
    print(f"\n  皮尔逊相关 r(实力差, 回合数) = {rho:+.3f}  （样本 {len(pairs)}）")
    print("  负值 = 实力越接近、回合数越多，符合直觉；绝对值小说明只是弱趋势")

# ---------------------------------------------------------------- 4
print()
print("=" * 74)
print("4) 「第 1 局胜者赢整场」——局中预测的参考线")
print("=" * 74)
# 按场聚合，取每场第 1 局（map_seq 最小的那局）
by_match = defaultdict(list)
for r in rows:
    by_match[r["matchid"]].append(r)
first = m1 = same = 0
for mid, ms in by_match.items():
    ms.sort(key=lambda x: x["map_seq"])
    g1 = ms[0]
    if g1["winner"] is None:
        continue
    first += 1
    a_won_first = (g1["winner"] == g1["teamA"])
    # 整场 A 是否赢：看 match 表
    mrow = c.execute("SELECT winner, teamA FROM match WHERE matchid=?", (mid,)).fetchone()
    if not mrow or mrow["winner"] is None:
        continue
    a_won_match = (mrow["winner"] == mrow["teamA"])
    if a_won_first == a_won_match:
        same += 1
print(f"  可判定场次：{first}")
print(f"  「第 1 局胜者 = 整场胜者」命中率：{same/first*100:.1f}%  ({same}/{first})")
print("  说明：这条规则只在『已经打完第 1 局』之后可用，属于局中信息，")
print("        跟赛前预测（mapseq=0）无关 —— 赛前必须换一套独立判断。")

c.close()
