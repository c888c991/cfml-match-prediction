# -*- coding: utf-8 -*-
"""sample_audit.py —— 样本量 / 阵容稳定性 / 时效性 审计

回答三个问题：
  1. 现在库里的数据，够不够训练一个预测模型？
  2. 用「队名」当预测对象有多危险？（阵容漂移有多快）
  3. 老数据该打几折？（时间衰减的半衰期怎么定）

用法：
  python sample_audit.py
"""
import os
import sqlite3
import sys
from collections import defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(BASE, "cfml.db")

# 只统计正式比赛
OFFICIAL = "is_official=1 AND is_placeholder=0"


def conn():
    if not os.path.exists(DB):
        sys.exit(f"找不到 {DB}，先跑 python build_db.py")
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    return c


def sec(title):
    print()
    print("=" * 74)
    print(title)
    print("=" * 74)


def season_order(c):
    """按实际开赛时间给赛季排序（contest 表本身没有时间）"""
    rows = c.execute(f"""
        SELECT contestid, contestname,
               MIN(match_time) first_time, MAX(match_time) last_time,
               COUNT(*) n_match, SUM(has_detail) n_detail
        FROM match WHERE {OFFICIAL}
        GROUP BY contestid, contestname
        ORDER BY first_time
    """).fetchall()
    return rows


# ---------------------------------------------------------------- 1. 总量
def part1(c, seasons):
    sec("一、数据总量")
    print(f'{"赛季":<18}{"对阵":>6}{"有明细":>7}{"覆盖率":>8}   {"起":<11}{"止":<11}')
    tot = tot_d = 0
    for s in seasons:
        cov = f"{s['n_detail'] / s['n_match'] * 100:.0f}%" if s["n_match"] else "-"
        print(f'{s["contestname"]:<18}{s["n_match"]:>6}{s["n_detail"] or 0:>7}{cov:>8}   '
              f'{(s["first_time"] or "")[:10]:<11}{(s["last_time"] or "")[:10]:<11}')
        tot += s["n_match"]
        tot_d += s["n_detail"] or 0
    print(f'{"合计":<18}{tot:>6}{tot_d:>7}{tot_d / tot * 100:>7.0f}%')

    n_map = c.execute("SELECT COUNT(*) FROM map").fetchone()[0]
    n_pm = c.execute("SELECT COUNT(*) FROM player_map").fetchone()[0]
    n_pl = c.execute("SELECT COUNT(DISTINCT player) FROM player_season "
                     "WHERE player IS NOT NULL").fetchone()[0]
    n_tm = c.execute(f"""
        SELECT COUNT(*) FROM (
            SELECT teamA t FROM match WHERE {OFFICIAL} AND teamA IS NOT NULL
            UNION SELECT teamB FROM match WHERE {OFFICIAL} AND teamB IS NOT NULL
        )""").fetchone()[0]
    print(f"\n单局(地图)记录 {n_map} 局")
    print(f"选手单局记录 {n_pm} 条")
    print(f"选手 {n_pl} 人   战队 {n_tm} 支")


# ------------------------------------------------- 2. 真正能用来训练的场次
def part2(c, seasons):
    sec("二、真正「能用来训练」的场次有多少")
    print("一场比赛要能进训练集，必须满足：① 双方都已有历史可算特征 ② 阵容归属可信")
    print("下面按时间顺序滚动，统计开赛前双方各有 >= N 场明细的场次。\n")

    rows = c.execute(f"""
        SELECT m.matchid, m.match_time, m.teamA, m.teamB, m.has_detail,
               m.contestname, m.n_maps, m.scoreA, m.scoreB
        FROM match m
        WHERE {OFFICIAL} AND m.teamA IS NOT NULL AND m.teamB IS NOT NULL
        ORDER BY m.match_time
    """).fetchall()

    seen = defaultdict(int)      # 队伍 -> 已打完且有明细的场次
    for need in (1, 3, 5, 10):
        for r in rows:
            pass
    # 一次遍历同时统计各阈值
    seen = defaultdict(int)
    cnt = {1: 0, 3: 0, 5: 0, 10: 0}
    detail_cnt = 0
    for r in rows:
        if not r["teamA"] or not r["teamB"]:
            continue
        a, b = r["teamA"], r["teamB"]
        if r["has_detail"]:
            detail_cnt += 1
            for need in cnt:
                if seen[a] >= need and seen[b] >= need:
                    cnt[need] += 1
            seen[a] += 1
            seen[b] += 1
    print(f'{"可用条件":<40}{"可训练场次":>10}')
    print(f'{"双方各 >= 1 场历史":<40}{cnt[1]:>10}')
    print(f'{"双方各 >= 3 场历史":<40}{cnt[3]:>10}')
    print(f'{"双方各 >= 5 场历史":<40}{cnt[5]:>10}')
    print(f'{"双方各 >= 10 场历史":<40}{cnt[10]:>10}')
    print(f'\n（对照：有逐局明细的比赛共 {detail_cnt} 场）')
    print("\n注：这是「单场胜负」的样本量。若预测「单局胜负」，样本翻约 2.5 倍。")


# ------------------------------------------------------ 3. 阵容漂移（核心）
def part3(c, seasons):
    sec("三、阵容漂移：「队名」这个身份到底有多稳定（核心）")
    print("用 player_season（12 赛季都可信）取每队每赛季出场最多的 5 人当主力阵容，")
    print("再算相邻赛季的重合度。重合度低 = 同一支队名的比赛其实是不同的人在打。\n")

    order = [s["contestid"] for s in seasons]
    idx = {cid: i for i, cid in enumerate(order)}

    # 每队每赛季的主力 5 人
    roster = defaultdict(dict)     # team -> contestid -> [players]
    rows = c.execute("""
        SELECT contestid, team, player, rounds
        FROM player_season
        WHERE player IS NOT NULL AND team IS NOT NULL
    """).fetchall()
    tmp = defaultdict(list)
    for r in rows:
        tmp[(r["team"], r["contestid"])].append((r["player"], r["rounds"] or 0))
    for (team, cid), lst in tmp.items():
        lst.sort(key=lambda x: -x[1])
        roster[team][cid] = set(p for p, _ in lst[:5])

    # 相邻赛季重合度
    print(f'{"战队":<12}{"赛季数":>6}  相邻赛季主力重合度（新赛季 vs 上一赛季，5 人）')
    all_j = []
    for team in sorted(roster):
        cids = sorted(roster[team], key=lambda x: idx.get(x, 999))
        if len(cids) < 2:
            continue
        pairs = []
        for i in range(1, len(cids)):
            prev, cur = roster[team][cids[i - 1]], roster[team][cids[i]]
            if not prev or not cur:
                continue
            j = len(prev & cur) / 5
            pairs.append(j)
            all_j.append(j)
        bar = " ".join(f"{int(j*5)}/5" for j in pairs)
        print(f"{team:<12}{len(cids):>6}  {bar}")
    if all_j:
        avg = sum(all_j) / len(all_j)
        print(f"\n单赛季主力平均保留 {avg * 5:.2f} / 5 人（保留率 {avg * 100:.0f}%）")
        # 推算：跨 k 个赛季后的期望重合
        print("\n按这个保留率推算，「同一队名」跨赛季后还剩多少人相同：")
        print(f'{"时间距离":<16}{"期望重合(人)":>14}')
        for k in (1, 2, 3, 4, 6, 8):
            print(f"{str(k) + ' 个赛季':<16}{5 * (avg ** k):>14.2f}")

    # 跨全局的重合：最早的赛季 vs 最新的赛季
    print("\n【极端对照】最早赛季的主力 vs 最新赛季的主力，还有几个人一样：")
    newest = order[-1]
    for team in sorted(roster):
        oldest_cid = next((x for x in sorted(roster[team], key=lambda y: idx.get(y, 999))), None)
        if oldest_cid is None or oldest_cid not in roster[team]:
            continue
        o, n = roster[team][oldest_cid], roster[team].get(newest, set())
        if not n:
            continue
        same = o & n
        so = next((s["contestname"] for s in seasons if s["contestid"] == oldest_cid), "?")
        print(f"   {team:<12} {so} 主力 vs 2026秋 主力： 相同 {len(same)}/5 人"
              + (f"  {sorted(same)}" if same else ""))


# ------------------------------------------------------ 4. 老将该不该打折
def part4(c, seasons):
    sec("四、选手状态随时间怎么变（用 KD 走势实证）")
    order = [s["contestid"] for s in seasons]
    idx = {cid: i for i, cid in enumerate(order)}
    name_of = {s["contestid"]: s["contestname"] for s in seasons}

    rows = c.execute("""
        SELECT contestid, player, team, rounds, kd
        FROM player_season
        WHERE player IS NOT NULL AND kd IS NOT NULL AND rounds >= 100
    """).fetchall()

    hist = defaultdict(dict)
    for r in rows:
        hist[r["player"]][r["contestid"]] = r["kd"]

    # 生涯 >= 4 个赛季、且出场回合数足够的选手
    vets = {p: h for p, h in hist.items() if len(h) >= 4}
    print(f"生涯横跨 >= 4 个赛季、且每季出场回合 >= 100 的选手：{len(vets)} 人\n")
    print(f'{"选手":<10}{"首季":<12}{"末季":<12}{"季数":>5}   KD 按时间顺序')
    up = down = flat = 0
    for p, h in sorted(vets.items(), key=lambda x: -len(x[1])):
        seq = [(idx[cid], cid, v) for cid, v in h.items()]
        seq.sort()
        kds = [v for _, _, v in seq]
        # 首尾比较：涨还是跌
        delta = kds[-1] - kds[0]
        if delta > 0.08:
            trend = "↑"
            up += 1
        elif delta < -0.08:
            trend = "↓"
            down += 1
        else:
            trend = "→"
            flat += 1
        s_ = " ".join(f"{v:.2f}" for v in kds)
        print(f'{p:<10}{name_of[seq[0][1]]:<12}{name_of[seq[-1][1]]:<12}'
              f'{len(kds):>5}  {s_}  {trend}')

    tot = up + down + flat
    if tot:
        print(f"\n首季 KD → 末季 KD：上升 {up} 人 ({up/tot*100:.0f}%)、"
              f"下滑 {down} 人 ({down/tot*100:.0f}%)、基本持平 {flat} 人 ({flat/tot*100:.0f}%)")
        print("（注意：老赛季的选手名归属可能有错，这里只用 player_season 的赛季累计 KD，")
        print("  它 12 赛季都可信；但仍是「赛季平均」，受对手强度影响。）")


# ------------------------------------------------------ 4.5 数据多旧会失效
def part4b(c, seasons):
    sec("四之二、「数据有多旧」对预测能力的量化（直接回答：早期数据该不该用）")
    order = [s["contestid"] for s in seasons]
    idx = {cid: i for i, cid in enumerate(order)}

    # ---------- (a) 选手 KD：用 k 个赛季前的数据预测本赛季，误差多大
    rows = c.execute("""
        SELECT contestid, player, kd FROM player_season
        WHERE player IS NOT NULL AND kd IS NOT NULL AND rounds >= 100
    """).fetchall()
    hist = defaultdict(dict)
    for r in rows:
        hist[r["player"]][r["contestid"]] = r["kd"]

    all_kd = [v for h in hist.values() for v in h.values()]
    mean_kd = sum(all_kd) / len(all_kd)
    sd = (sum((v - mean_kd) ** 2 for v in all_kd) / len(all_kd)) ** 0.5

    print("【A. 选手 KD】用「k 个赛季前的 KD」去推测「本赛季 KD」，平均误差：\n")
    print(f'{"用多久前的数据":<18}{"平均绝对误差":>14}{"相对基线":>12}')
    print(f'{"拿平均值瞎猜（基线）":<18}{sd:>14.3f}{"100%":>12}')
    for lag in (1, 2, 3, 4, 6, 8):
        errs = []
        for p, h in hist.items():
            seq = sorted(h.items(), key=lambda x: idx.get(x[0], 999))
            for i in range(len(seq)):
                if i - lag >= 0:
                    errs.append(abs(seq[i][1] - seq[i - lag][1]))
        if errs:
            mae = sum(errs) / len(errs)
            print(f'{"上一赛季" if lag == 1 else str(lag) + " 个赛季前":<18}'
                  f'{mae:>14.3f}{mae / sd * 100:>11.0f}%   (n={len(errs)})')
    print("\n读法：数字越小越准。越接近基线 100% = 这条旧数据已经没有信息量，等于没用。")

    # ---------- (b) 队伍胜率：用 k 个赛季前的胜率预测本赛季
    print("\n【B. 队伍胜率】用「k 个赛季前的胜率」推测「本赛季胜率」，相关系数：\n")
    trows = c.execute(f"""
        SELECT m.contestid, m.teamA t, m.scoreA s, m.scoreB s2, m.matchid
        FROM match m WHERE {OFFICIAL} AND m.has_detail=1 AND m.teamA IS NOT NULL
        UNION ALL
        SELECT m.contestid, m.teamB, m.scoreB, m.scoreA, m.matchid
        FROM match m WHERE {OFFICIAL} AND m.has_detail=1 AND m.teamB IS NOT NULL
    """).fetchall()
    win = defaultdict(lambda: [0, 0])       # (team, contestid) -> [胜, 总]
    for r in trows:
        k = (r["t"], r["contestid"])
        win[k][1] += 1
        if r["s"] is not None and r["s"] > r["s2"]:
            win[k][0] += 1
    rate = {k: v[0] / v[1] for k, v in win.items() if v[1] >= 5}

    def pearson(xs, ys):
        n = len(xs)
        if n < 3:
            return None
        mx, my = sum(xs) / n, sum(ys) / n
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        dx = sum((x - mx) ** 2 for x in xs) ** 0.5
        dy = sum((y - my) ** 2 for y in ys) ** 0.5
        return num / (dx * dy) if dx and dy else None

    print(f'{"用多久前的胜率":<18}{"相关系数":>10}{"样本对":>8}')
    for lag in (1, 2, 3, 4, 6, 8):
        xs, ys = [], []
        for (team, cid), v in rate.items():
            i = idx.get(cid)
            if i is None:
                continue
            prev = None
            for (t2, cid2), v2 in rate.items():
                if t2 == team and idx.get(cid2) == i - lag:
                    prev = v2
            if prev is not None:
                xs.append(prev)
                ys.append(v)
        r_ = pearson(xs, ys)
        if r_ is not None:
            print(f'{"上一赛季" if lag == 1 else str(lag) + " 个赛季前":<18}'
                  f'{r_:>10.2f}{len(xs):>8}')
    print("\n读法：1.00 = 完全一致（旧数据照样准）；0.00 = 毫无关系（旧数据完全没用）。")
    print("      一个赛季 ≈ 半年，一次转会窗平均换掉 1/4 阵容。")


# ------------------------------------------------------ 5. 折算成可用样本
def part5(c, seasons):
    sec("五、时间窗口一收紧，样本还剩多少")
    print("假设「只用最近 N 天的比赛」，看看还剩多少场（以最新一场比赛为基准日）\n")
    last = c.execute(f"SELECT MAX(match_time) FROM match WHERE {OFFICIAL}").fetchone()[0]
    print(f"最新一场比赛时间：{last}\n")
    print(f'{"窗口":<12}{"有明细场次":>12}{"占比":>8}   覆盖赛季')
    total = c.execute(f"SELECT COUNT(*) FROM match WHERE {OFFICIAL} AND has_detail=1").fetchone()[0]
    for days in (90, 180, 365, 730, 1095, None):
        if days is None:
            n = total
            tag = "全部"
            seasons_txt = "12 个赛季"
        else:
            n = c.execute(f"""
                SELECT COUNT(*) FROM match
                WHERE {OFFICIAL} AND has_detail=1
                  AND julianday(?) - julianday(match_time) <= ?
            """, (last, days)).fetchone()[0]
            tag = f"近 {days} 天"
            names = [r[0] for r in c.execute(f"""
                SELECT DISTINCT contestname FROM match
                WHERE {OFFICIAL} AND has_detail=1
                  AND julianday(?) - julianday(match_time) <= ?
                ORDER BY contestname
            """, (last, days))]
            seasons_txt = "、".join(names) if names else "-"
        pct = f"{n / total * 100:.0f}%" if total else "-"
        print(f"{tag:<12}{n:>12}{pct:>8}   {seasons_txt}")


def main():
    c = conn()
    seasons = season_order(c)
    part1(c, seasons)
    part2(c, seasons)
    part3(c, seasons)
    part4(c, seasons)
    part4b(c, seasons)
    part5(c, seasons)
    c.close()
    print()


if __name__ == "__main__":
    main()
