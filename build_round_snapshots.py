# -*- coding: utf-8 -*-
"""build_round_snapshots.py —— 从原始 JSON 抽出「局中快照」表

## 为什么需要它

之前以为历史数据里没有局中状态（比如「第 1 局打到 5:3 时」）。
实际上每个 game_*.json 里，每局都带一份逐回合流水：
    round = {"1":{"win_way":"击杀","win_team":"EP"}, "2":{...}, ...}
把它按顺序累加，就还原出「第 r 回合结束时比分 a:b」。

**唯一限制**：这份流水最多只存 16 个回合。加时局（总数 17~28 回合）只存了前 16 回合。
不过常规时间占 51%，足够支撑「局中实时预测」的建模。

## 产出

表 round_snap：每个 (比赛, 局, 回合) 一个快照点
    matchid, map_seq, round_no        —— 这是第几局、打到第几回合
    cur_a, cur_b                      —— 此刻比分（A=数据库 map.teamA 那一方）
    final_a, final_b                  —— 这一局的最终比分
    a_wins_map                        —— 这一局 A 是否赢
    flow_complete                     —— 流水是否完整覆盖到终局（False=加时局被截断）
    match_time, contestname           —— 便于按时间切分

用法：python build_round_snapshots.py
"""
import json
import os
import sqlite3
from collections import Counter

BASE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(BASE, "raw")
DB = os.path.join(BASE, "cfml.db")

SCHEMA = """
DROP TABLE IF EXISTS round_snap;
CREATE TABLE round_snap(
  matchid INTEGER, map_seq INTEGER, round_no INTEGER,
  cur_a INTEGER, cur_b INTEGER,
  final_a INTEGER, final_b INTEGER,
  a_wins_map INTEGER,
  flow_complete INTEGER,
  match_time TEXT, contestname TEXT,
  PRIMARY KEY (matchid, map_seq, round_no)
);
CREATE INDEX idx_rs_time ON round_snap(match_time);
"""


def main():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    c.executescript(SCHEMA)

    # 数据库里「有明细」的场次 + 每局最终比分（teamA 为 A 方）
    maps = {}
    meta = {}
    for r in c.execute("""
        SELECT m.matchid, m.map_seq, m.teamA_raw, m.scoreA, m.scoreB,
               mt.match_time, mt.contestname
        FROM map m JOIN match mt ON mt.matchid=m.matchid
        WHERE mt.is_official=1 AND mt.is_placeholder=0 AND mt.has_detail=1
    """):
        maps[(r["matchid"], r["map_seq"])] = r
        meta[r["matchid"]] = (r["match_time"], r["contestname"])

    by_match = {}
    for (mid, mseq), r in maps.items():
        by_match.setdefault(mid, {})[mseq] = r

    n_snap = 0
    n_maps = 0
    n_complete = 0
    n_truncated = 0
    no_file = 0
    name_miss = 0
    round_dist = Counter()

    for mid, mseq_map in by_match.items():
        fp = os.path.join(RAW, f"game_{mid}.json")
        if not os.path.exists(fp):
            no_file += 1
            continue
        try:
            data = (json.load(open(fp, encoding="utf-8")) or {}).get("data") or {}
        except Exception:
            no_file += 1
            continue
        if not isinstance(data, dict):
            continue

        for mseq, mr in mseq_map.items():
            mv = data.get(str(mseq))
            if not isinstance(mv, dict):
                continue
            rd = mv.get("round")
            if not (isinstance(rd, dict) and rd):
                continue
            ta = (mv.get("data", {}).get("teamA", {}).get("team", {}) or {}).get("team_name")
            tb = (mv.get("data", {}).get("teamB", {}).get("team", {}) or {}).get("team_name")
            # 数据库里 A 方的原始队名，用来把流水归到 A/B
            db_a = mr["teamA_raw"]

            rounds = sorted(int(k) for k in rd if str(k).isdigit())
            if not rounds:
                continue
            cur_a = cur_b = 0
            fa = mr["scoreA"] or 0
            fb = mr["scoreB"] or 0
            # 判断 JSON 的 A 队是不是数据库的 A 队（有时顺序会反）
            a_is_db_a = (ta == db_a)
            complete = (len(rounds) >= fa + fb)
            if complete:
                n_complete += 1
            else:
                n_truncated += 1
            n_maps += 1
            round_dist[len(rounds)] += 1

            for rno in range(1, min(16, len(rounds)) + 1):
                w = (rd.get(str(rno)) or {}).get("win_team")
                if w == ta:
                    if a_is_db_a:
                        cur_a += 1
                    else:
                        cur_b += 1
                elif w == tb:
                    if a_is_db_a:
                        cur_b += 1
                    else:
                        cur_a += 1
                else:
                    name_miss += 1
                    continue
                c.execute("""INSERT OR REPLACE INTO round_snap(
                    matchid,map_seq,round_no,cur_a,cur_b,final_a,final_b,
                    a_wins_map,flow_complete,match_time,contestname
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?)""", (
                    mid, mseq, rno, cur_a, cur_b, fa, fb,
                    1 if fa > fb else 0, 1 if complete else 0,
                    mr["match_time"], mr["contestname"],
                ))
                n_snap += 1

    c.commit()
    print("=" * 72)
    print("局中快照表构建完成")
    print("=" * 72)
    print(f"  覆盖单局：{n_maps}  其中流水完整 {n_complete}，加时被截断 {n_truncated}")
    print(f"  快照点：{n_snap}")
    print(f"  读不到原始文件的对阵：{no_file}   认不出队名的回合：{name_miss}")
    print(f"  （校验）快照点应等于 round_snap 行数："
          f"{c.execute('SELECT COUNT(*) FROM round_snap').fetchone()[0]}")
    c.close()


if __name__ == "__main__":
    main()
