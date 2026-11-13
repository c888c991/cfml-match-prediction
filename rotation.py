# -*- coding: utf-8 -*-
"""
选手轮换 / 转会 专项分析

回答四件事：
  1. 阵容     —— 每个赛季每支队伍有几个人上过场，谁主力谁替补
  2. 中途换人 —— 同一场比赛里，不同局之间换了人的情况
  3. 转会     —— 跨赛季换队的选手名单
  4. 队名变更 —— 同一支队伍在不同赛季的名字演变（做统计时必须先归一化）

用法：
    python rotation.py              # 全部
    python rotation.py lineup       # 只看阵容
    python rotation.py switch       # 只看中途换人
    python rotation.py transfer     # 只看转会
    python rotation.py alias        # 只看队名变更
"""
import os
import sqlite3
import sys
from collections import defaultdict

DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cfml.db")

# 只有这两个赛季的「选手↔队伍」归属是可信的（详见 README 开头）。
# 更早的赛季，接口把选手名替换成了"当前阵容"，会造成假的转会和假的轮换。
RELIABLE = ["2026CFML夏季赛", "2026CFML秋季赛"]


def conn():
    if not os.path.exists(DB):
        sys.exit(f"找不到数据库 {DB}，先跑 python build_db.py")
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    return c


def warn():
    print("!" * 72)
    print("数据口径说明（详见 README.md 开头那节）：")
    print("  · 阵容与转会 → 用赛季名单接口，12 个赛季都可信 ✓")
    print("  · 逐局换人   → 用比赛明细，只有 2026 夏、2026 秋可靠")
    print("                 （更早赛季接口把选手名替换成了『当前阵容』）")
    print("!" * 72)
    print()


USE_ALL = False


def season_where(contest=None, col="c.name", reliable_only=False):
    """构造赛季过滤条件"""
    conds, args = [], []
    if reliable_only and not USE_ALL:
        conds.append(f"{col} IN ({','.join('?' * len(RELIABLE))})")
        args += RELIABLE
    if contest:
        conds.append(f"{col} LIKE ?")
        args.append(f"%{contest}%")
    return (" AND ".join(conds) if conds else "1=1"), args


# ------------------------------------------------------------ 1. 阵容
def lineup(c, contest=None):
    print("=" * 72)
    print("【阵容】每赛季每队上场人数与出场回合分布")
    print("=" * 72)
    print("说明：rounds = 该选手本赛季出场的回合数（来自赛季名单接口，归属可信）。")
    print("同队里明显偏低的就是替补。\n")

    q = """SELECT c.name season, ps.team, ps.team_raw, ps.player, ps.position, ps.rounds
           FROM player_season ps JOIN contest c ON c.contestid=ps.contestid"""
    cond, args = season_where(contest)
    q += f" WHERE {cond}"

    by = defaultdict(lambda: defaultdict(list))
    for r in c.execute(q, args):
        by[r["season"]][r["team"] or "(未标注)"].append(
            (r["player"], r["rounds"] or 0, r["position"] or ""))

    for season in sorted(by):
        print(f"── {season} " + "─" * (60 - len(season)))
        for team in sorted(by[season]):
            lst = sorted(by[season][team], key=lambda x: -x[1])
            hi = lst[0][1] if lst else 0
            core = [x for x in lst if x[1] >= hi * 0.6] if hi else lst
            subs = [x for x in lst if x not in core]
            print(f"  {team:<12} {len(lst)} 人报名")
            for p, rnd, pos in core:
                print(f"        主力 {p:<10} {pos:<4} {rnd:>5} 回合")
            for p, rnd, pos in subs:
                print(f"        替补 {p:<10} {pos:<4} {rnd:>5} 回合")
        print()


# ------------------------------------------------------------ 2. 中途换人
def switch(c, contest=None):
    print("=" * 72)
    print("【中途换人】同一场比赛内，不同局之间上场选手发生变化")
    print("=" * 72)
    print("说明：一场比赛如果某队各局上场的人不一样，就是局间换人。")
    print("      只有 2026 夏、2026 秋的逐局选手名可信。\n")

    where = "is_official=1 AND has_detail=1 AND is_placeholder=0"
    args = []
    if contest:
        where += " AND contestname LIKE ?"
        args.append(f"%{contest}%")
    if not USE_ALL:
        where += f" AND contestname IN ({','.join('?' * len(RELIABLE))})"
        args += RELIABLE

    matches = c.execute(
        f"SELECT matchid, contestname, teamA, teamB, match_time, n_maps "
        f"FROM match WHERE {where} AND n_maps>1 ORDER BY match_time", args).fetchall()

    hit = 0
    for m in matches:
        rows = c.execute("""SELECT map_seq, team, player FROM player_map
                            WHERE matchid=? AND player IS NOT NULL
                            ORDER BY map_seq""", (m["matchid"],)).fetchall()
        per = defaultdict(lambda: defaultdict(set))
        for r in rows:
            per[r["team"]][r["map_seq"]].add(r["player"])
        for team, seqs in per.items():
            allp = set().union(*seqs.values())
            if len(allp) > 5:
                hit += 1
                who = sorted(allp)
                detail = " | ".join(
                    f"第{s}局: {'/'.join(sorted(seqs[s]))}" for s in sorted(seqs))
                print(f"  ⚑ {m['match_time'][:16]}  {m['teamA']} vs {m['teamB']}  [{m['contestname']}]")
                print(f"      {team} 上场 {len(allp)} 人：{', '.join(who)}")
                print(f"      {detail}")
                print()

    print(f"共 {len(matches)} 场多局比赛，其中 {hit} 次出现局间换人。")


# ------------------------------------------------------------ 3. 转会
def transfer(c):
    print("=" * 72)
    print("【转会】跨赛季更换队伍的选手")
    print("=" * 72)
    print("数据源：赛季名单接口（归属可信，与官方转会公告一致）。\n")

    rows = c.execute("""
        SELECT ps.player, ps.team, ps.team_raw, c.name season, c.contestid
        FROM player_season ps JOIN contest c ON c.contestid=ps.contestid
        WHERE ps.player IS NOT NULL AND ps.team IS NOT NULL
        ORDER BY c.contestid""").fetchall()

    hist = defaultdict(dict)
    for r in rows:
        hist[r["player"]][r["contestid"]] = r["team"]

    moved = {p: h for p, h in hist.items() if len(set(h.values())) > 1}
    print(f"共 {len(hist)} 名选手有归属记录，其中 {len(moved)} 人换过队伍。\n")

    for p, h in sorted(moved.items(), key=lambda x: (-len(set(x[1].values())), x[0])):
        seq = [h[cid] for cid in sorted(h)]
        print(f"  {p:<10} " + " → ".join(seq))
    return hist, moved


# ------------------------------------------------------------ 4. 队名变更
def alias(c):
    print("=" * 72)
    print("【队名变更】同一支队伍在不同赛季的名字")
    print("=" * 72)
    print("说明：team_raw 是接口原样返回的名字，team 是归一化后的规范名。\n")

    rows = c.execute("""
        SELECT team, contestname, team_raw
        FROM player_season ps JOIN contest c ON c.contestid=ps.contestid
        GROUP BY team, contestname, team_raw
        ORDER BY team, c.contestid""").fetchall()

    by = defaultdict(dict)
    for r in rows:
        by[r["team"]][r["contestname"]] = r["team_raw"]

    for team in sorted(by):
        names = sorted({v for v in by[team].values() if v})
        if len(names) > 1:
            print(f"  {team:<12} 曾用名：{'、'.join(names)}")
    print()
    print("  上面这些如果不归一化，统计时会被当成不同队伍，胜负记录会碎成好几份。")


def main():
    global USE_ALL
    argv = [a for a in sys.argv[1:] if a != "--all"]
    USE_ALL = "--all" in sys.argv
    cmd = argv[0] if argv else "all"
    arg = argv[1] if len(argv) > 1 else None
    c = conn()
    warn()
    if cmd in ("all", "lineup"):
        lineup(c, arg)
    if cmd in ("all", "switch"):
        switch(c, arg)
    if cmd in ("all", "transfer"):
        transfer(c)
    if cmd in ("all", "alias"):
        alias(c)
    c.close()


if __name__ == "__main__":
    main()
