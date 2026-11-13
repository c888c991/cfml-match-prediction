# -*- coding: utf-8 -*-
"""
CFML 数据库查询命令行

用法：
    python query.py seasons                     各赛季规模
    python query.py teams [赛季关键词]           战队一览（胜率、场均KD）
    python query.py matches <赛季关键词>          某赛季赛程与比分
    python query.py player <选手名> [赛季关键词]   某选手逐局明细
    python query.py head2head <队A> <队B>        两队历史交手
    python query.py team <队名> [赛季关键词]      某战队逐场战绩
    python query.py upcoming [赛季关键词]        还没打的比赛（赛程已公布、明细还没数据）
    python query.py search <关键词>              搜索选手 / 战队
"""
import os
import sqlite3
import sys

DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cfml.db")


def conn():
    if not os.path.exists(DB):
        sys.exit(f"找不到数据库 {DB}，先跑 python build_db.py")
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    return c


def table(rows, cols, headers=None, widths=None):
    if not rows:
        print("  （无结果）")
        return
    headers = headers or cols
    data = [[("" if r[c] is None else str(r[c])) for c in cols] for r in rows]
    widths = widths or [max(len(headers[i]), *(len(d[i]) for d in data)) for i in range(len(cols))]
    line = "  ".join(headers[i].ljust(widths[i]) for i in range(len(cols)))
    print(line)
    print("-" * len(line))
    for d in data:
        print("  ".join(d[i].ljust(widths[i]) for i in range(len(cols))))


def cmd_seasons(c, *a):
    rows = c.execute("""
        SELECT t.name AS 赛季,
               COUNT(DISTINCT m.matchid) AS 对阵,
               SUM(CASE WHEN m.has_detail=1 THEN 1 ELSE 0 END) AS 有明细,
               COUNT(pm.id) AS 选手记录
        FROM contest t
        LEFT JOIN match m ON m.contestid=t.contestid AND m.is_official=1
        LEFT JOIN player_map pm ON pm.contestid=t.contestid
        GROUP BY t.contestid ORDER BY t.contestid""").fetchall()
    table(rows, ["赛季", "对阵", "有明细", "选手记录"])


def cmd_teams(c, key=None):
    where, args = "is_official=1 AND is_placeholder=0", []
    if key:
        where += " AND m.contestname LIKE ?"
        args.append(f"%{key}%")
    rows = c.execute(f"""
        SELECT team AS 战队, COUNT(DISTINCT m.matchid) AS 场次,
               COUNT(DISTINCT player) AS 选手数,
               ROUND(AVG(kd),2) AS 场均KD
        FROM player_map pm JOIN match m ON m.matchid=pm.matchid
        WHERE {where} GROUP BY team ORDER BY 场次 DESC""", args).fetchall()
    table(rows, ["战队", "场次", "选手数", "场均KD"])


def cmd_matches(c, key):
    rows = c.execute("""
        SELECT m.match_time AS 时间, m.teamA AS A, m.teamB AS B,
               COALESCE(m.scoreA||':'||m.scoreB,'') AS 比分,
               COALESCE(m.winner,'') AS 胜者, m.n_maps AS 局数,
               m.subcontestname AS 轮次
        FROM match m WHERE m.is_official=1 AND m.is_placeholder=0
          AND m.contestname LIKE ? ORDER BY m.match_time""", (f"%{key}%",)).fetchall()
    table(rows, ["时间", "A", "B", "比分", "胜者", "局数", "轮次"])


def cmd_team(c, name, key=None):
    where, args = "pm.team=?", [name]
    if key:
        where += " AND m.contestname LIKE ?"
        args.append(f"%{key}%")
    rows = c.execute(f"""
        SELECT m.match_time AS 时间, m.teamA AS A, m.teamB AS B,
               m.scoreA AS sA, m.scoreB AS sB, m.winner AS 胜者,
               m.contestname AS 赛季
        FROM match m JOIN player_map pm ON pm.matchid=m.matchid
        WHERE {where} GROUP BY m.matchid ORDER BY m.match_time""", args).fetchall()
    table(rows, ["时间", "A", "B", "sA", "sB", "胜者", "赛季"])
    if rows:
        w = sum(1 for r in rows if r["胜者"] == name)
        print(f"\n  {name}：{len(rows)} 场，{w} 胜，胜率 {w/len(rows)*100:.1f}%")


def cmd_player(c, name, key=None):
    where, args = "player=?", [name]
    if key:
        where += " AND contestid IN (SELECT contestid FROM contest WHERE name LIKE ?)"
        args.append(f"%{key}%")
    rows = c.execute(f"""
        SELECT m.match_time AS 时间, pm.team AS 战队, pm.position AS 位置,
               pm.map_seq AS 局, pm.kill AS K, pm.death AS D, pm.kd AS KD,
               pm.damage AS 伤害, pm.headshot_rate AS 爆头率,
               m.teamA||' vs '||m.teamB AS 对阵
        FROM player_map pm JOIN match m ON m.matchid=pm.matchid
        WHERE {where} ORDER BY m.match_time, pm.map_seq""", args).fetchall()
    table(rows, ["时间", "战队", "位置", "局", "K", "D", "KD", "伤害", "爆头率", "对阵"])
    if rows:
        import statistics
        kds = [r["KD"] for r in rows if r["KD"] is not None]
        print(f"\n  {name}：出场 {len(rows)} 局，平均 KD {statistics.mean(kds):.2f}，"
              f"最高 {max(kds):.2f}，最低 {min(kds):.2f}")


def cmd_head2head(c, a, b):
    rows = c.execute("""
        SELECT m.match_time AS 时间, m.contestname AS 赛季,
               m.teamA AS A, m.teamB AS B, m.scoreA AS sA, m.scoreB AS sB,
               m.winner AS 胜者
        FROM match m WHERE m.is_official=1 AND m.is_placeholder=0
          AND ((m.teamA=? AND m.teamB=?) OR (m.teamA=? AND m.teamB=?))
        ORDER BY m.match_time""", (a, b, b, a)).fetchall()
    table(rows, ["时间", "赛季", "A", "B", "sA", "sB", "胜者"])
    if rows:
        wa = sum(1 for r in rows if r["胜者"] == a)
        wb = sum(1 for r in rows if r["胜者"] == b)
        print(f"\n  历史交手 {len(rows)} 次：{a} {wa} 胜 — {b} {wb} 胜")


def cmd_upcoming(c, key=None, limit="40"):
    """还没打的比赛。这些场次赛程里已经有了（连 matchid 都有），
    只是明细接口还没数据 —— 正好拿来做事前预测。"""
    where = "m.is_official=1 AND m.is_placeholder=0 AND m.has_detail=0"
    args = []
    if key:
        where += " AND m.contestname LIKE ?"
        args.append(f"%{key}%")
    rows = c.execute(f"""
        SELECT m.match_time AS 时间, m.contestname AS 赛季,
               m.teamA AS A, m.teamB AS B, m.subcontestname AS 轮次, m.matchid AS matchid
        FROM match m WHERE {where}
        ORDER BY m.match_time LIMIT ?""", args + [int(limit)]).fetchall()
    table(rows, ["时间", "赛季", "A", "B", "轮次", "matchid"])
    if rows:
        print(f"\n  以上 {len(rows)} 场已在赛程里，但还没打完。")
        print("  跑 python fetch_raw.py --update 会把新打完的场次数据补进来。")


def cmd_search(c, key):
    print("【选手】")
    for r in c.execute("""SELECT player, GROUP_CONCAT(DISTINCT team) teams,
                                 COUNT(*) n FROM player_map WHERE player LIKE ?
                          GROUP BY player ORDER BY n DESC LIMIT 20""", (f"%{key}%",)):
        print(f"   {r[0]:<10} {r[2]:>4} 局   {r[1]}")
    print("\n【战队】")
    for r in c.execute("""SELECT team, team_raw, COUNT(*) n FROM player_map
                          WHERE team LIKE ? OR team_raw LIKE ?
                          GROUP BY team ORDER BY n DESC LIMIT 20""", (f"%{key}%", f"%{key}%")):
        print(f"   {r[0]:<12} raw={r[1]:<12} {r[2]} 条")


CMDS = {
    "seasons": cmd_seasons, "teams": cmd_teams, "matches": cmd_matches,
    "team": cmd_team, "player": cmd_player, "head2head": cmd_head2head,
    "upcoming": cmd_upcoming, "search": cmd_search,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in CMDS:
        print(__doc__)
        sys.exit(0)
    cmd = CMDS[sys.argv[1]]
    try:
        cmd(conn(), *sys.argv[2:])
    except TypeError:
        print(f"参数不对。\n{__doc__}")
