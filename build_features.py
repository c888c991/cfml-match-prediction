# -*- coding: utf-8 -*-
"""build_features.py —— 生成「预测点」特征表

## 什么是一个预测点

一场比赛不是一个样本，而是一串样本。以 BO3 为例：

    map_seq = 0   开赛前        大比分 0:0
    map_seq = 1   打完第 1 局后  1:0 或 0:1
    map_seq = 2   打完第 2 局后  2:0 / 1:1 / 0:2

每个点带两个标签：

    y_match   本场最终 A 是否获胜（整场 -> 适合「赛前 + 局间」预测）
    y_map     下一局 A 是否获胜（单局 -> 适合贴着实时比分的预测）

## 铁律：所有赛前特征只能用「该场开赛之前」的比赛算

本脚本按时间顺序处理比赛，每场算完特征才把结果写进历史。
顺序一旦错，回测分数就是假的。

## 为什么没有「当前局比分」这个特征

历史数据里的每一局都是「已打完」的状态，没有「第 1 局打到 5:3 时」这种中间快照。
所以模型训练不出来「局中比分 -> 结果」的映射。
实时场景下「当前局比分」只能靠规则/贝叶斯修正，或者等实时数据攒够了再单独训一个局内模型。

用法：
    python build_features.py                 # 默认半衰期 365 天
    python build_features.py --halflife 180  # 换半衰期
    python build_features.py --csv           # 顺便导出 CSV 方便肉眼检查
"""
import argparse
import csv
import os
import sqlite3
import sys
from collections import defaultdict

# 卡尔曼实力跟踪复用内核的实现 —— 别再写第二份（selfcheck 的检查 1 就是查这件事）。
from cfml_core import kalman_strength, kalman_feats, is_playoff_of

BASE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(BASE, "cfml.db")

HALFLIFE = 365.0     # 时间衰减半衰期（天）。0 或负数 = 不衰减，用简单平均。
MIN_DAYS = 1.0       # 衰减用的最小天数差，防止除零


def conn():
    if not os.path.exists(DB):
        sys.exit(f"找不到 {DB}，先跑 python build_db.py")
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    return c


# ------------------------------------------------------------------ 工具
def days_between(t1, t2):
    """两个 'YYYY-MM-DD HH:MM:SS' 之间差多少天（t2 - t1）"""
    from datetime import datetime
    f = "%Y-%m-%d %H:%M:%S"
    try:
        a = datetime.strptime(t1[:19], f)
        b = datetime.strptime(t2[:19], f)
    except Exception:
        return None
    return (b - a).total_seconds() / 86400.0


def weighted(pairs, halflife):
    """pairs = [(权重基准天数, 值)]，返回衰减加权平均。halflife<=0 时退化为简单平均。"""
    if not pairs:
        return None
    if halflife is None or halflife <= 0:
        return sum(v for _, v in pairs) / len(pairs)
    num = den = 0.0
    for age, v in pairs:
        w = 0.5 ** (max(age, 0.0) / halflife)
        num += w * v
        den += w
    return num / den if den else None


# ------------------------------------------------------------------ 数据装载
def load_matches(c):
    return c.execute("""
        SELECT m.matchid, m.contestid, m.contestname, m.subcontestname, m.match_time,
               m.teamA, m.teamB, m.n_maps, m.winner, m.scoreA, m.scoreB
        FROM match m
        WHERE m.is_official=1 AND m.is_placeholder=0 AND m.has_detail=1
          AND m.teamA IS NOT NULL AND m.teamB IS NOT NULL
          AND m.match_time IS NOT NULL
        ORDER BY m.match_time, m.matchid
    """).fetchall()


def load_maps(c):
    """matchid -> [该场每局的信息]，按局序号排好"""
    d = defaultdict(list)
    for r in c.execute("""
        SELECT matchid, map_seq, teamA, teamB, scoreA, scoreB, killA, killB, winner
        FROM map ORDER BY matchid, map_seq
    """):
        d[r["matchid"]].append(dict(r))
    return d


def load_deaths(c):
    """(matchid, map_seq, team) -> 该方 5 人的死亡数之和（用来算队伍 KD）"""
    d = defaultdict(int)
    for r in c.execute("""
        SELECT matchid, map_seq, team, SUM(death) d
        FROM player_map WHERE team IS NOT NULL
        GROUP BY matchid, map_seq, team
    """):
        d[(r["matchid"], r["map_seq"], r["team"])] = r["d"] or 0
    return d


# ------------------------------------------------------------------ 特征表结构
SCHEMA = """
DROP TABLE IF EXISTS feature_point;
CREATE TABLE feature_point(
  matchid INTEGER,
  contestid INTEGER,
  contestname TEXT,
  mapseq INTEGER,              -- 0=开赛前，j=已打完 j 局
  match_time TEXT,
  teamA TEXT, teamB TEXT,
  n_maps INTEGER,              -- 本场一共打了(或要打)几局
  is_playoff INTEGER,
  y_match INTEGER,             -- 本场最终 A 是否获胜
  y_map INTEGER,               -- 下一局 A 是否获胜（最后一局后为 NULL）
  -- 赛前：近期状态（时间衰减加权）
  a_wr REAL, b_wr REAL,                 -- 胜率
  a_mapwr REAL, b_mapwr REAL,           -- 单局胜率
  a_netmaps REAL, b_netmaps REAL,       -- 平均净胜局
  a_rd REAL, b_rd REAL,                 -- 平均回合分差
  a_n REAL, b_n REAL,                   -- 有效历史场次（衰减后）
  a_sznwr REAL, b_sznwr REAL,           -- 本赛季胜率（开赛前）
  h2h_wr REAL, h2h_n INTEGER,           -- 交手：A 的胜率、交手场次
  h2h_last INTEGER,                     -- 上次交手 A 是否赢
  -- 卡尔曼实力跟踪：替代上面 a_wr/b_wr 与 a_n/b_n 的「手工半衰期」口径
  -- 由 cfml_core.kalman_strength 产出（EKF，状态=队伍实力，见 exp_kalman.py）
  kf_diff REAL,                         -- 实力差 x_A − x_B
  kf_conf REAL,                         -- 不确定性差（log 精度差，越大=越确定）
  -- 赛中：已打完的那几局累积出来的
  score_diff INTEGER,                   -- 当前大比分差（A - B）
  rnd_diff INTEGER,                     -- 已打局回合分差（A - B）
  a_kd REAL, b_kd REAL,                 -- 已打局队伍 KD
  last_map_won INTEGER,                 -- 上一局 A 是否赢（mapseq=0 时 NULL）
  close_maps INTEGER,                   -- 已打局里分差 <=2 的局数（势均力敌程度）
  PRIMARY KEY (matchid, mapseq)
);
CREATE INDEX idx_fp_time ON feature_point(match_time);
"""


def build(halflife=HALFLIFE, export_csv=False):
    c = conn()
    matches = load_matches(c)
    maps_of = load_maps(c)
    deaths = load_deaths(c)

    c.executescript(SCHEMA)

    # 卡尔曼实力跟踪：一场一场滚动，取每场「开赛前」的状态（天然无泄漏）
    kf, _ = kalman_strength(matches)

    hist = defaultdict(list)        # team -> [{time, won, maps_won, maps_lost, rf, ra, kd}]
    h2h = defaultdict(list)         # (A,B) 规范化后的键 -> [A 是否赢]
    season_games = defaultdict(list)   # (team, contestid) -> [是否赢]

    n_points = 0
    n_skipped = 0

    for m in matches:
        a, b = m["teamA"], m["teamB"]
        t = m["match_time"]
        cid = m["contestid"]
        if a == b:
            n_skipped += 1
            continue

        maps = maps_of.get(m["matchid"]) or []
        if not maps:
            n_skipped += 1
            continue

        n_maps = len(maps)
        is_playoff = is_playoff_of(m["subcontestname"])

        # ---------------- 赛前特征：只用 t 之前的比赛 ----------------
        def pre(team):
            past = [x for x in hist[team] if x["time"] < t]
            if not past:
                return dict(wr=None, mapwr=None, netmaps=None, rd=None, n=0.0)
            age = lambda x: max(days_between(x["time"], t) or 0.0, MIN_DAYS)
            w_wr = weighted([(age(x), x["won"]) for x in past], halflife)
            w_map = weighted([(age(x), x["mw"] / x["nm"] if x["nm"] else 0.5) for x in past], halflife)
            w_net = weighted([(age(x), x["mw"] - x["ml"]) for x in past], halflife)
            w_rd = weighted([(age(x), x["rf"] - x["ra"]) for x in past], halflife)
            # 衰减后的"有效场次"：越久远贡献越小
            if halflife and halflife > 0:
                eff = sum(0.5 ** (age(x) / halflife) for x in past)
            else:
                eff = float(len(past))
            return dict(wr=w_wr, mapwr=w_map, netmaps=w_net, rd=w_rd, n=eff)

        fa, fb = pre(a), pre(b)

        # 本赛季胜率（同样只用开赛前的）
        def szn(team):
            past = [x for x in season_games[(team, cid)] if x[0] < t]
            return (sum(v for _, v in past) / len(past)) if past else None

        sa, sb = szn(a), szn(b)

        # 交手记录（对称存储）
        key = tuple(sorted((a, b)))
        pair = [x for x in h2h[key] if x["time"] < t]
        if pair:
            a_is_first = key[0] == a
            wins_a = sum(1 for x in pair if (x["a_won"] if a_is_first else not x["a_won"]))
            h2h_wr = wins_a / len(pair)
            h2h_n = len(pair)
            last = pair[-1]
            h2h_last = 1 if (last["a_won"] if a_is_first else not last["a_won"]) else 0
        else:
            h2h_wr, h2h_n, h2h_last = None, 0, None

        # ---------------- 逐局推进，生成每个预测点 ----------------
        y_match = 1 if m["winner"] == a else 0

        cum_rf = cum_ra = 0      # A / B 的回合分累计
        cum_ka = cum_da = cum_kb = cum_db = 0
        score_a = score_b = 0
        close_maps = 0
        last_map_won = None

        # 预测点 j = 「已打完 j 局」，最多到 n_maps-1（最后一局还没打）。
        # 千万别写 range(0, n_maps+1) —— 那会多出一个「全部打完」的快照点，
        # 那时比分已定、标签必然正确，模型会从这个点学到假规律，回测分数虚高一大截。
        for j in range(0, n_maps):
            nxt = maps[j]                     # 即将打的那一局，此刻还没打
            y_map = 1 if nxt["winner"] == a else 0

            rnd_diff = cum_rf - cum_ra
            a_kd = (cum_ka / cum_da) if cum_da else None
            b_kd = (cum_kb / cum_db) if cum_db else None

            kfd, kfc = kalman_feats(kf, m["matchid"])
            c.execute("""INSERT OR REPLACE INTO feature_point(
                matchid, contestid, contestname, mapseq, match_time, teamA, teamB,
                n_maps, is_playoff, y_match, y_map,
                a_wr, b_wr, a_mapwr, b_mapwr, a_netmaps, b_netmaps, a_rd, b_rd,
                a_n, b_n, a_sznwr, b_sznwr, h2h_wr, h2h_n, h2h_last,
                kf_diff, kf_conf,
                score_diff, rnd_diff, a_kd, b_kd, last_map_won, close_maps
            ) VALUES(""" + ",".join("?" * 34) + ")", (
                m["matchid"], cid, m["contestname"], j, t, a, b, n_maps, is_playoff,
                y_match, y_map,
                fa["wr"], fb["wr"], fa["mapwr"], fb["mapwr"],
                fa["netmaps"], fb["netmaps"], fa["rd"], fb["rd"],
                fa["n"], fb["n"],
                sa, sb, h2h_wr, h2h_n, h2h_last,
                kfd, kfc,
                score_a - score_b, rnd_diff, a_kd, b_kd, last_map_won, close_maps,
            ))
            n_points += 1

            # 打完第 j 局，把结果并进累积，供下一个预测点使用
            g = nxt
            sa_, sb_ = g["scoreA"] or 0, g["scoreB"] or 0
            ka_, kb_ = g["killA"] or 0, g["killB"] or 0
            da_ = deaths.get((m["matchid"], g["map_seq"], a), 0)
            db_ = deaths.get((m["matchid"], g["map_seq"], b), 0)
            cum_rf += sa_
            cum_ra += sb_
            cum_ka += ka_
            cum_da += da_
            cum_kb += kb_
            cum_db += db_
            if g["winner"] == a:
                score_a += 1
            elif g["winner"] == b:
                score_b += 1
            if abs(sa_ - sb_) <= 2:
                close_maps += 1
            last_map_won = 1 if g["winner"] == a else 0

        # ---------------- 把本场结果写进历史 ----------------
        a_won = (m["winner"] == a)
        a_mw = sum(1 for g in maps if g["winner"] == a)
        b_mw = sum(1 for g in maps if g["winner"] == b)
        a_rf = sum((g["scoreA"] or 0) for g in maps)
        a_ra = sum((g["scoreB"] or 0) for g in maps)
        a_k = sum((g["killA"] or 0) for g in maps)
        a_d = sum(deaths.get((m["matchid"], g["map_seq"], a), 0) for g in maps)

        rec_a = dict(time=t, won=1 if a_won else 0, mw=a_mw, ml=b_mw, nm=n_maps,
                     rf=a_rf, ra=a_ra, kd=(a_k / a_d if a_d else None))
        rec_b = dict(time=t, won=0 if a_won else 1, mw=b_mw, ml=a_mw, nm=n_maps,
                     rf=a_ra, ra=a_rf, kd=None)
        hist[a].append(rec_a)
        hist[b].append(rec_b)
        season_games[(a, cid)].append((t, 1 if a_won else 0))
        season_games[(b, cid)].append((t, 0 if a_won else 1))

        h2h[key].append({"time": t, "a_won": a_won})

    c.commit()

    # ---------------- 报告 ----------------
    q = lambda s: c.execute(s).fetchone()[0]
    print("=" * 72)
    print("特征表构建完成")
    print("=" * 72)
    print(f"  比赛 {len(matches)} 场（跳过 {n_skipped} 场无逐局数据/自身对阵）")
    print(f"  预测点 {n_points} 个（含开赛前的 {q('SELECT COUNT(*) FROM feature_point WHERE mapseq=0')} 个）")
    print(f"  可用标签：y_match {q('SELECT COUNT(*) FROM feature_point WHERE y_match IS NOT NULL')} 个，"
          f"y_map {q('SELECT COUNT(*) FROM feature_point WHERE y_map IS NOT NULL')} 个")
    print(f"  赛前特征齐全（双方都有历史）的点："
          f"{q('SELECT COUNT(*) FROM feature_point WHERE a_n>0 AND b_n>0')} 个")
    print(f"  半衰期 {halflife} 天")

    rows = c.execute("""
        SELECT COUNT(*) n, AVG(y_match) wr FROM feature_point
        WHERE mapseq=0 AND a_n>0 AND b_n>0
    """).fetchone()
    if rows and rows["n"]:
        print(f"\n  对照基线（开赛前，只统计双方都有历史的点）：")
        print(f"    样本 {rows['n']} 个，A 的整体胜率 {rows['wr']*100:.1f}%"
              f"（A 是赛程里排在前面的一方，没有主场含义，接近 50% 说明赛程排列无偏）")

    if export_csv:
        out = os.path.join(BASE, "features.csv")
        cur = c.execute("SELECT * FROM feature_point")
        cols = [d[0] for d in cur.description]
        with open(out, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(cols)
            w.writerows(cur.fetchall())
        print(f"\n  已导出 {out}")

    c.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--halflife", type=float, default=HALFLIFE,
                    help="时间衰减半衰期（天），0 = 不衰减")
    ap.add_argument("--csv", action="store_true", help="顺便导出 features.csv")
    args = ap.parse_args()
    build(halflife=args.halflife, export_csv=args.csv)
