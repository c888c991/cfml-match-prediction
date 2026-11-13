# -*- coding: utf-8 -*-
"""
CFML 数据建库：把 raw/ 里的原始 JSON 解析进本地 SQLite

用法：
    python build_db.py            # 建库（覆盖重建）
    python build_db.py --report   # 建库并打印数据体检报告

设计要点：
- 队名 / 选手名会跨赛季变化（R.LGD斗鱼 → LGD、KZ → KZ.斗鱼），
  建库时同时保留 raw 名和归一化名。预测建模一律用归一化名。
- 每局只记录真正上场的 5 人（接口本身就是这个粒度），
  所以"轮换/替补"不用额外处理，数据天然带着。
"""
import json
import os
import sqlite3
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")
DB = os.path.join(HERE, "cfml.db")

# ---------------------------------------------------------------- 队名归一化
# 同一个战队在不同赛季/不同赞助冠名下的写法，归到同一个规范名。
# 规范名统一用 2026 秋季赛的叫法。
ALIAS = {
    # LGD
    "R.LGD": "LGD", "R.LGD斗鱼": "LGD", "LGD": "LGD", "lgd": "LGD",
    # KZ
    "KZ": "KZ.斗鱼", "KZ.斗鱼": "KZ.斗鱼", "KZ斗鱼": "KZ.斗鱼",
    # 白鲨
    "白鲨": "白鲨", "白鲨.斗鱼": "白鲨", "白鲨斗鱼": "白鲨",
    "BS": "白鲨", "白鲨DY": "白鲨",
    # AG
    "AG": "成都AG", "AG.虎牙": "成都AG", "AG虎牙": "成都AG",
    "成都AG": "成都AG", "成都AG.虎牙": "成都AG",
    # Q9 / 情久
    "Q9": "Q9.快手", "Q9.快手": "Q9.快手", "Q9快手": "Q9.快手",
    "情久": "Q9.快手", "情久.快手": "Q9.快手", "情久快手": "Q9.快手",
    # XROCK
    "XROCK": "XROCK.快手", "XROCK.B站": "XROCK.快手", "XROCK.快手": "XROCK.快手",
    "XROCK快手": "XROCK.快手",
    # EP
    "EP": "EP.快手", "EP.快手": "EP.快手", "EP快手": "EP.快手",
    # eStar
    "eStar": "eStar", "estar": "eStar", "ESTAR": "eStar", "EStar": "eStar",
    # 其他（存在过但已退出）
    "WE": "WE", "we": "WE",
    "TLG": "TLG", "EDG": "EDG",
    "EDGM": "EDGM",          # EDG 二队，独立统计
    "SN": "SN", "Sn": "SN",
    "待定": "待定",           # 队伍未定，靠 is_placeholder 标记排除
}

_unknown_teams = defaultdict(int)

# ---------------------------------------------------------------- 战队分组
# 有些"改名"其实是同一支队伍换了品牌，统计时要不要合并由你决定。
# 目前只合并 WE → TLG（2026 秋季赛起 WE 的整套主力出现在 TLG 名下，赛场内无 WE）。
# 想拆开就把下面的映射删掉。
TEAM_GROUP = {
    "WE": "WE/TLG",
    "TLG": "WE/TLG",
}


def team_group(team):
    return TEAM_GROUP.get(team, team)


def norm_team(name):
    if name is None:
        return None
    n = str(name).strip()
    if n in ALIAS:
        return ALIAS[n]
    _unknown_teams[n] += 1
    return n


def norm_player(full_name, team=None):
    """选手名归一化：去掉 '.队名' / '队名' 后缀，只留 ID。
    接口里同一人写法不一：'热恋.KZ斗鱼' vs 'KZ.斗鱼'。
    """
    if not full_name:
        return None
    s = str(full_name).strip()
    if "." in s:
        head, _, tail = s.partition(".")
        # 后缀是队名（含点的情况如 KZ.斗鱼 会被切成 KZ / 斗鱼，只取头部即可）
        return head.strip()
    # 没有点：尝试从尾部剥掉归一化过或原始的队名
    cands = set()
    if team:
        cands.add(str(team))
        cands.add(norm_team(team) or "")
    for t in list(ALIAS.keys()):
        cands.add(t)
    for t in sorted(cands, key=len, reverse=True):
        if t and s.endswith(t) and len(s) > len(t):
            return s[: -len(t)].strip()
    return s


def num(v, cast=float):
    if v is None or v == "":
        return None
    try:
        return cast(v)
    except Exception:
        return None


# ---------------------------------------------------------------- 建表
SCHEMA = """
DROP TABLE IF EXISTS contest;
DROP TABLE IF EXISTS subcontest;
DROP TABLE IF EXISTS match;
DROP TABLE IF EXISTS map;
DROP TABLE IF EXISTS player_map;
DROP TABLE IF EXISTS player_season;
DROP TABLE IF EXISTS team_season;

CREATE TABLE contest(
  contestid INTEGER PRIMARY KEY,
  name TEXT
);

CREATE TABLE subcontest(
  subcontestid INTEGER PRIMARY KEY,
  contestid INTEGER,
  name TEXT,
  start_time TEXT,
  end_time TEXT
);

CREATE TABLE match(
  matchid INTEGER PRIMARY KEY,
  contestid INTEGER,
  contestname TEXT,
  subcontestid INTEGER,
  subcontestname TEXT,
  seq_in_sub INTEGER,
  teamA_raw TEXT, teamB_raw TEXT,
  teamA TEXT, teamB TEXT,
  match_time TEXT,
  is_official INTEGER DEFAULT 1,     -- 0 = 全明星/表演赛/待定，建模时要排除
  is_placeholder INTEGER DEFAULT 0,  -- 1 = 队伍还没定（"待定"）
  has_detail INTEGER DEFAULT 0,
  n_maps INTEGER DEFAULT 0,
  winner TEXT,
  scoreA INTEGER, scoreB INTEGER
);

CREATE TABLE map(
  map_uid TEXT PRIMARY KEY,          -- matchid + 局序号
  matchid INTEGER,
  map_seq INTEGER,
  contestid INTEGER,
  teamA TEXT, teamB TEXT,
  teamA_raw TEXT, teamB_raw TEXT,
  scoreA INTEGER, scoreB INTEGER,
  killA INTEGER, killB INTEGER,
  winner TEXT,
  map_name TEXT                      -- 原始地图名（如"新潜艇"）；2021 赛季约半数缺失
);

CREATE TABLE player_map(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  map_uid TEXT,
  matchid INTEGER,
  map_seq INTEGER,
  contestid INTEGER,
  team TEXT, team_raw TEXT, team_group TEXT,
  player TEXT, player_raw TEXT,
  position TEXT,
  kill INTEGER, death INTEGER,
  kd REAL, damage INTEGER,
  round_kill REAL, round_damage REAL,
  first_kill INTEGER, first_kill_rate TEXT,
  survival_rate REAL, hit_rate TEXT, headshot_rate REAL,
  three_kill INTEGER, four_kill INTEGER, five_kill INTEGER
);

CREATE TABLE player_season(
  contestid INTEGER,
  contestname TEXT,
  player TEXT, player_raw TEXT,
  team TEXT, team_raw TEXT,
  position TEXT,
  rounds INTEGER,
  kill INTEGER, death INTEGER, kd REAL,
  round_damage REAL, round_kill REAL,
  rounds_won_pct TEXT,
  mvp INTEGER,
  PRIMARY KEY (contestid, player, team)
);

CREATE TABLE team_season(
  contestid INTEGER,
  contestname TEXT,
  team TEXT, team_raw TEXT,
  team_json TEXT
);
CREATE INDEX idx_pm_map ON player_map(map_uid);
CREATE INDEX idx_pm_player ON player_map(player);
CREATE INDEX idx_map_match ON map(matchid);
CREATE INDEX idx_m_match ON match(contestid, subcontestid);
"""


def load(name):
    p = os.path.join(RAW, name)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def main():
    report = "--report" in sys.argv
    if os.path.exists(DB):
        os.remove(DB)
    conn = sqlite3.connect(DB)
    conn.executescript(SCHEMA)

    contests = load("_contests.json") or []
    index = load("_index.json") or []
    cname = {c["contestid"]: c["contestname"] for c in contests}

    for c in contests:
        conn.execute("INSERT OR REPLACE INTO contest VALUES(?,?)",
                     (c["contestid"], c["contestname"]))
    conn.commit()
    print(f"赛季 {len(contests)} 个，对阵索引 {len(index)} 条")

    # ---------------- 轮次
    for c in contests:
        subs = load(f'sub_{c["contestid"]}.json') or []
        for s in subs:
            conn.execute("INSERT OR REPLACE INTO subcontest VALUES(?,?,?,?,?)",
                         (s["subcontestid"], c["contestid"], s["subcontestname"],
                          s.get("conteststarttime"), s.get("contestendtime")))
    conn.commit()

    # ---------------- 对阵（只统计有 matchid 的）
    seen = set()
    rows = []
    for x in index:
        mid = x.get("matchid")
        if not mid or mid in seen:
            continue
        seen.add(mid)
        rows.append(x)
    print(f"去重后对阵 {len(rows)} 场")

    # 同一轮次内的顺序
    seq = defaultdict(int)
    n_allstar = n_placeholder = 0
    for x in rows:
        key = (x["contestid"], x["subcontestid"])
        seq[key] += 1
        sn = x["subcontestname"] or ""
        official = 0 if ("全明星" in sn or "单挑" in sn or "表演" in sn) else 1
        placeholder = 1 if (x.get("teamA") == "待定" or x.get("teamB") == "待定") else 0
        if not official:
            n_allstar += 1
        if placeholder:
            n_placeholder += 1
        # 全明星表演赛的队名可能是选手 ID，不参与归一化（反正要排除）
        ta = norm_team(x["teamA"]) if official else x["teamA"]
        tb = norm_team(x["teamB"]) if official else x["teamB"]
        conn.execute(
            "INSERT OR REPLACE INTO match(matchid,contestid,contestname,subcontestid,"
            "subcontestname,seq_in_sub,teamA_raw,teamB_raw,teamA,teamB,match_time,"
            "is_official,is_placeholder) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (x["matchid"], x["contestid"], x["contestname"], x["subcontestid"],
             x["subcontestname"], seq[key], x["teamA"], x["teamB"],
             ta, tb, x["time"],
             official, placeholder))
    conn.commit()
    print(f"其中：全明星/表演赛 {n_allstar} 场，队伍待定 {n_placeholder} 场")

    # ---------------- 逐局明细
    n_map = n_pm = n_detail = 0
    for x in rows:
        mid = x["matchid"]
        r = load(f"game_{mid}.json")
        if not r:
            continue
        d = (r.get("data") or {})
        keys = [k for k in d.keys() if str(k).strip() != ""]
        if not keys:
            continue                      # 老赛季无明细
        n_detail += 1
        winA = winB = 0
        maps = []
        for k in keys:
            try:
                mseq = int(k)
            except Exception:
                continue
            body = (d[k] or {}).get("data") or {}
            map_name = (d[k] or {}).get("map_name")
            ta, tb = body.get("teamA"), body.get("teamB")
            if not ta or not tb:
                continue
            tA, tB = ta.get("team") or {}, tb.get("team") or {}
            sa, sb = num(tA.get("score"), int), num(tB.get("score"), int)
            ka, kb = num(tA.get("kill"), int), num(tB.get("kill"), int)
            uid = f"{mid}_{mseq}"
            ma_team_raw, mb_team_raw = tA.get("team_name"), tB.get("team_name")
            if (sa or 0) > (sb or 0):
                winA += 1
                winner = norm_team(ma_team_raw)
            elif (sb or 0) > (sa or 0):
                winB += 1
                winner = norm_team(mb_team_raw)
            else:
                winner = None
            conn.execute(
                "INSERT OR REPLACE INTO map(map_uid,matchid,map_seq,contestid,"
                "teamA,teamB,teamA_raw,teamB_raw,scoreA,scoreB,killA,killB,winner,map_name) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (uid, mid, mseq, x["contestid"],
                 norm_team(ma_team_raw), norm_team(mb_team_raw),
                 ma_team_raw, mb_team_raw, sa, sb, ka, kb, winner,
                 (map_name or "").strip() or None))
            n_map += 1
            maps.append(uid)
            for side in (ta, tb):
                t = side.get("team") or {}
                t_raw = t.get("team_name")
                t_norm = norm_team(t_raw)
                for p in (side.get("player") or []):
                    p_raw = p.get("playerName")
                    conn.execute(
                        "INSERT INTO player_map(map_uid,matchid,map_seq,contestid,"
                        "team,team_raw,team_group,player,player_raw,position,kill,death,kd,damage,"
                        "round_kill,round_damage,first_kill,first_kill_rate,survival_rate,"
                        "hit_rate,headshot_rate,three_kill,four_kill,five_kill) "
                        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (uid, mid, mseq, x["contestid"], t_norm, t_raw, team_group(t_norm),
                         norm_player(p_raw, t_raw), p_raw, p.get("Position"),
                         num(p.get("kill"), int), num(p.get("death"), int),
                         num(p.get("KD"), float), num(p.get("damage"), int),
                         num(p.get("round_kill"), float), num(p.get("round_damage"), float),
                         num(p.get("first_kill"), int), p.get("first_kill_rate"),
                         num(p.get("survival_rate"), float), p.get("hit_rate"),
                         num(p.get("headshot_rate"), float),
                         num(p.get("three_kill"), int), num(p.get("four_kill"), int),
                         num(p.get("five_kill"), int)))
                    n_pm += 1
        wname = norm_team(x["teamA"]) if winA > winB else (norm_team(x["teamB"]) if winB > winA else None)
        conn.execute(
            "UPDATE match SET has_detail=1,n_maps=?,winner=?,scoreA=?,scoreB=? WHERE matchid=?",
            (len(maps), wname, winA, winB, mid))
    conn.commit()
    print(f"有逐局明细的对阵 {n_detail} 场，单局 {n_map} 局，选手单局记录 {n_pm} 条")

    # ---------------- 赛季选手（轮换 / 转会）
    n_ps = 0
    for c in contests:
        cid = c["contestid"]
        ps = load(f"_players_{cid}.json") or []
        for p in ps:
            full = p.get("player") or ""
            t_raw = p.get("team")
            conn.execute(
                "INSERT OR REPLACE INTO player_season("
                "contestid,contestname,player,player_raw,team,team_raw,position,"
                "rounds,kill,death,kd,round_damage,round_kill,rounds_won_pct,mvp) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (cid, c["contestname"], norm_player(full, t_raw), full,
                 norm_team(t_raw), t_raw, p.get("position"),
                 num(p.get("round"), int), num(p.get("kill"), int), num(p.get("death"), int),
                 num(p.get("kd"), float), num(p.get("round_damage"), float),
                 num(p.get("round_kill"), float), p.get("round_win_rate"),
                 num(p.get("mvp"), int)))
            n_ps += 1
        ts = load(f"_teams_{cid}.json") or []
        for t in ts:
            t_raw = t.get("team") or t.get("teamName") or t.get("name")
            conn.execute("INSERT INTO team_season(contestid,contestname,team,team_raw,team_json) "
                         "VALUES(?,?,?,?,?)",
                         (cid, c["contestname"], norm_team(t_raw), t_raw,
                          json.dumps(t, ensure_ascii=False)))
    conn.commit()
    print(f"赛季选手记录 {n_ps} 条")

    conn.close()

    if _unknown_teams:
        print()
        print("=" * 60)
        print("⚠ 队名归一化表里有未收录的写法（建议补进 ALIAS）：")
        for k, v in sorted(_unknown_teams.items(), key=lambda x: -x[1]):
            print(f"   {k}  ×{v}")
        print("=" * 60)

    if report:
        verify(DB)


def verify(db):
    conn = sqlite3.connect(db)
    q = lambda s: conn.execute(s).fetchall()
    print()
    print("=" * 64)
    print("数据体检报告")
    print("=" * 64)
    print("\n【按赛季】（只算正式比赛）")
    print(f'{"赛季":<20}{"对阵":>6}{"有明细":>7}{"单局":>6}{"选手记录":>8}')
    for cid, name in q("SELECT contestid,name FROM contest ORDER BY contestid"):
        n_m = q(f"SELECT COUNT(*) FROM match WHERE contestid={cid} AND is_official=1")[0][0]
        n_d = q(f"SELECT COUNT(*) FROM match WHERE contestid={cid} AND is_official=1 AND has_detail=1")[0][0]
        n_mp = q(f"SELECT COUNT(*) FROM map WHERE contestid={cid}")[0][0]
        n_p = q(f"SELECT COUNT(*) FROM player_map WHERE contestid={cid}")[0][0]
        if n_m == 0:
            continue
        print(f"{name:<20}{n_m:>6}{n_d:>7}{n_mp:>6}{n_p:>8}")

    print("\n【总计】")
    for label, sql in [
        ("赛季", "SELECT COUNT(*) FROM contest"),
        ("对阵（全部）", "SELECT COUNT(*) FROM match"),
        ("对阵（正式）", "SELECT COUNT(*) FROM match WHERE is_official=1"),
        ("有逐局明细的对阵", "SELECT COUNT(*) FROM match WHERE has_detail=1"),
        ("单局（地图）", "SELECT COUNT(*) FROM map"),
        ("选手单局记录", "SELECT COUNT(*) FROM player_map"),
        ("不同选手（归一化后）", "SELECT COUNT(DISTINCT player) FROM player_map"),
        ("不同战队（归一化后）", "SELECT COUNT(DISTINCT team) FROM player_map"),
    ]:
        print(f"   {label:<22}{q(sql)[0][0]}")

    print("\n【数据完整度】选手名字缺失情况（越早的赛季缺得越多）")
    print(f'{"赛季":<20}{"选手记录":>8}{"有名字":>8}{"具名率":>8}')
    for cid, name in q("SELECT contestid,name FROM contest ORDER BY contestid"):
        tot = q(f"SELECT COUNT(*) FROM player_map WHERE contestid={cid}")[0][0]
        if not tot:
            continue
        named = q(f"SELECT COUNT(*) FROM player_map WHERE contestid={cid} AND player IS NOT NULL")[0][0]
        print(f"{name:<20}{tot:>8}{named:>8}{named/tot*100:>7.0f}%")

    print("\n【出场最多的选手 Top 15】")
    for p, n, kd in q("""SELECT player, COUNT(*) n, ROUND(AVG(kd),2)
                         FROM player_map WHERE player IS NOT NULL
                         GROUP BY player ORDER BY n DESC LIMIT 15"""):
        print(f"   {p:<10} 出场 {n:>4} 局   平均KD {kd}")

    print("\n【轮换证据：同队内出场局数差异最大的 12 组】")
    rows = q("""
      SELECT team, COUNT(DISTINCT player) np, MIN(cnt), MAX(cnt)
      FROM (SELECT team, player, COUNT(*) cnt FROM player_map
            WHERE player IS NOT NULL GROUP BY team, player) GROUP BY team HAVING np>1
      ORDER BY (MAX(cnt)-MIN(cnt)) DESC LIMIT 12""")
    for team, np_, lo, hi in rows:
        print(f"   {team:<12} {np_} 人，出场局数 {lo} ~ {hi}")

    print("\n【转会证据：效力过 2 支以上战队的选手】")
    rows = q("""SELECT player, COUNT(DISTINCT team) nt,
                       GROUP_CONCAT(DISTINCT team) teams
                FROM player_map WHERE player IS NOT NULL
                GROUP BY player HAVING nt>1
                ORDER BY nt DESC LIMIT 12""")
    print(f"   合计 {q('''SELECT COUNT(*) FROM (SELECT player FROM player_map
             WHERE player IS NOT NULL GROUP BY player HAVING COUNT(DISTINCT team)>1)''')[0][0]} 人换过队")
    for p, nt, ts in rows:
        print(f"   {p:<10} {nt} 支：{ts}")
    conn.close()


if __name__ == "__main__":
    main()
