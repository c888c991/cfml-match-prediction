# -*- coding: utf-8 -*-
"""exp_lineup.py —— 选手阵容特征实验

## 想法来源

用户提出：选手短期不会换，所以可以用「上一场的阵容」代表这一场，
从而把**选手级信息**引进模型。这是当前最大的一块空白（模型一直只有队伍级特征）。

实测前提（probe_lineup*.py）：相邻两场阵容完全相同率 **85~92%**，换 ≥2 人几乎为零。
→ 代理成立。

## 选手名缺失这个坑

原始 JSON 里每队每局**恒为 5 人**，但部分选手的 playerName 是 null。
缺名率：2022~2023 约 25~30%，2024 约 16%，2025 约 13%，2026夏 6.7%，**2026秋 0%**。
所以越新的数据，选手级特征越完整。

## 三个方案（滚动前向，口径与 backtest.py 一致：WARMUP=80 / RETRAIN=50）

| | 说明 | 实盘可行性 |
|---|---|---|
| A 基线 | 现有 15 维（含卡尔曼实力差/不确定性差） | — |
| B 代理阵容 | A + 「上一场阵容」推出的选手级特征 | **直接可用**，不需要新数据 |
| C 真实阵容 | A + 本场真实阵容推出的选手级特征 | 需要从直播画面读阵容（上限参考） |

B 与 C 的差额 = 「读阵容」这件事值多少分。

用法：
    python exp_lineup.py              # 完整对比
    python exp_lineup.py --diag       # 只看诊断（代理阵容的命中率等）
"""
import argparse
import math
from collections import defaultdict

from cfml_core import (connect, load_all, featurize, sigmoid, weighted,
                       HALFLIFE, MIN_DAYS, parse_time, days_between)

WARMUP = 80      # 与 backtest.py 一致
RETRAIN = 50
EPOCHS, LR, L2 = 260, 0.5, 0.002   # 与 cfml_core.train_logreg 一致

# ---- 数据可信度闸门（用户定：检查如果有矛盾，就直接不用）----
# 判据：块内**具名**选手里，有多大比例真的在「该队当赛季名单」内（名单来自 player_season）。
# 不达标的块，名字不可信 → 这一块直接不进阵容、不进球员历史。
MIN_NAMED = 3      # 块内至少 3 个具名选手才够判断
MIN_CONS = 0.6     # 用户定的门槛：一致率 ≥ 60%

# 候选的选手级特征。关键区分：
#   rate / kd  —— 阵容的「水平」，与卡尔曼实力差高度相关（实测 0.81）→ 很可能是冗余的
#   stab / chg —— 阵容的「变动」，队伍级特征完全没有这个信息 → 真正的新东西
#   delta      —— 阵容质量的「变化量」（这一场代理阵容 vs 两场前），用来修正卡尔曼的滞后
LU_DOC = {
    "lu_rate_diff": "代理阵容平均个人能力差（每回合击杀数）",
    "lu_kd_diff": "代理阵容平均 KD 差",
    "lu_stab_diff": "阵容稳定度差（连续相同阵容的场次，封顶 10）",
    "lu_chg_diff": "近期是否换人差（1=上场阵容与再上一场不同）",
    "lu_delta_diff": "阵容质量变化差（代理阵容 vs 两场前阵容的个人能力差）",
    "tagg_rk_diff": "队内平均每回合击杀差（队伍聚合，不经选手名）",
    "tagg_rd_diff": "队内平均每回合伤害差（队伍聚合，不经选手名）",
    "tagg_kd_diff": "队伍 KD 差（队伍聚合，不经选手名）",
}


# ------------------------------------------------------------------ 工具
def fnum(v, default=None):
    """接口的数值常是字符串（'13' / '5%'），统一转 float。"""
    if v is None:
        return default
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().replace("%", "")
    if not s:
        return default
    try:
        return float(s)
    except ValueError:
        return default


# ------------------------------------------------------------------ 装载
def load_roster(c):
    """每个赛季每支队伍的名单（来自 player_season，12 赛季都可信）。

    用途：给比赛明细的「队块」做体检——块里的具名选手有多少比例真在当赛季名单内。
    """
    d = defaultdict(set)
    for r in c.execute("SELECT contestid, team, player FROM player_season "
                       "WHERE player IS NOT NULL AND team IS NOT NULL"):
        d[(r["contestid"], r["team"])].add(r["player"])
    return d


def load_team_agg(c):
    """每场每队的「队内平均个人指标」。

    ⚠️ 关键：这一路**不经过选手名**，直接用块内全部行算。
    实测（probe_lineup9/10）：老赛季每行的选手名是按「队伍现役阵容」套上去的（名字错），
    但**行本身是这场比赛的真人**（块内击杀之和 == 队伍总击杀，4630/4630 全赛季成立）。
    所以「队伍聚合」这一路是干净的、全赛季可用；「按名字区分到人」那一路不能用。
    """
    acc = defaultdict(lambda: {"n": 0, "rk": 0.0, "rd": 0.0, "k": 0.0, "d": 0.0})
    for r in c.execute("""SELECT matchid, team, round_kill, round_damage, kill, death
                          FROM player_map WHERE team IS NOT NULL"""):
        a = acc[(r["matchid"], r["team"])]
        a["n"] += 1
        a["rk"] += fnum(r["round_kill"], 0.0)
        a["rd"] += fnum(r["round_damage"], 0.0)
        a["k"] += fnum(r["kill"], 0.0)
        a["d"] += fnum(r["death"], 0.0)
    out = {}
    for k, a in acc.items():
        n = a["n"] or 1
        out[k] = {"rk": a["rk"] / n, "rd": a["rd"] / n,
                  "kd": a["k"] / a["d"] if a["d"] > 0 else None}
    return out


def _team_at(agg_hist, team, t, key):
    """队伍在时刻 t 的衰减加权队内平均指标（只用 t 之前的场次）"""
    h = [x for x in agg_hist.get(team, []) if x[0] < t and x[1].get(key) is not None]
    if not h:
        return None
    return weighted([(days_between(x[0], t) or MIN_DAYS, x[1][key]) for x in h], HALFLIFE)


def load_player_data(c):
    """读选手逐局数据，聚合成「每场每队每选手」一行。

    返回：
      lineup[(mid, team)]        = set(选手)         该场该队出场的具名选手
      stat[(mid, team, player)]  = dict(rk, rd, kd)  该场该选手的个人指标（三种口径）
    """
    agg = defaultdict(lambda: {"maps": 0, "kill": 0.0, "death": 0.0,
                               "rk": 0.0, "rd": 0.0})
    for r in c.execute("""
        SELECT matchid, team, player, kill, death, round_kill, round_damage
        FROM player_map WHERE player IS NOT NULL AND team IS NOT NULL
    """):
        k = (r["matchid"], r["team"], r["player"])
        a = agg[k]
        a["maps"] += 1
        a["kill"] += fnum(r["kill"], 0.0)
        a["death"] += fnum(r["death"], 0.0)
        a["rk"] += fnum(r["round_kill"], 0.0)
        a["rd"] += fnum(r["round_damage"], 0.0)

    lineup = defaultdict(set)
    stat = {}
    for (mid, team, player), a in agg.items():
        n = a["maps"] or 1
        lineup[(mid, team)].add(player)
        stat[(mid, team, player)] = {
            "rk": a["rk"] / n,                                    # 每回合击杀
            "rd": a["rd"] / n,                                    # 每回合伤害
            "kd": a["kill"] / a["death"] if a["death"] > 0 else None,
        }
    return lineup, stat


def load_points(c):
    """feature_point 的全部预测点（与 backtest.py 同一批）。"""
    rows = [dict(r) for r in c.execute("""
        SELECT * FROM feature_point
        WHERE y_match IS NOT NULL AND match_time IS NOT NULL
        ORDER BY match_time, matchid, mapseq
    """)]
    by_match = defaultdict(list)
    mids = []
    for r in rows:
        if r["matchid"] not in by_match:
            mids.append(r["matchid"])
        by_match[r["matchid"]].append(r)
    return mids, by_match


# ------------------------------------------------------------------ 阵容特征
def build_lineup_feats(c, rating="rk"):
    """按时间滚动，为每场比赛算出「代理阵容」和「真实阵容」两套特征。

    严格只用该场**之前**的比赛数据，天然无泄漏。
    rating = 个人能力口径：rk=每回合击杀 / rd=每回合伤害 / kd=击杀死亡比
    返回 {matchid: {...}}
    """
    ms, _, _ = load_all(c)
    lineup, stat = load_player_data(c)
    tagg = load_team_agg(c)
    roster = load_roster(c)               # (contestid, team) -> 当赛季名单
    agg_hist = defaultdict(list)          # team -> [(time, 队内平均指标)]

    # 每队按时间排好的比赛
    team_ms = defaultdict(list)
    for m in ms:
        for t in (m["teamA"], m["teamB"]):
            team_ms[t].append(m["matchid"])
    # ms 已按时间排序，所以 team_ms 天然有序

    player_hist = defaultdict(list)      # player -> [(time_str, 指标 dict)]
    last_lu = {}                          # team -> 上一场阵容
    prev_lu = {}                          # team -> 再上一场阵容
    stab = {}                             # team -> 连续相同阵容场次

    out = {}
    for m in ms:
        mid, t = m["matchid"], m["match_time"]
        rec = {}

        # --- 数据可信度闸门：这一场两个队块的名字对不对得上当赛季名单 ---
        trusted, cons = {}, {}
        for team in (m["teamA"], m["teamB"]):
            named = lineup.get((mid, team), set())
            ros = roster.get((m["contestid"], team), set())
            c_ = (len(named & ros) / len(named)) if named else 0.0
            cons[team] = c_
            trusted[team] = bool(len(named) >= MIN_NAMED and ros and c_ >= MIN_CONS)

        for side, team in (("A", m["teamA"]), ("B", m["teamB"])):
            actual = lineup.get((mid, team), set())
            proxy = last_lu.get(team, set())
            rec["actual_" + side] = actual
            rec["proxy_" + side] = proxy
            rec["trust_" + side] = 1 if trusted[team] else 0
            rec["cons_" + side] = cons[team]

            # --- 代理阵容特征 ---
            pr, pk, nk = [], [], 0
            for p in proxy:
                v = _player_at(player_hist, p, t, rating)
                if v is None:
                    continue
                nk += 1
                if v[0] is not None:
                    pr.append(v[0])
                if v[1] is not None:
                    pk.append(v[1])
            rec["proxy_rate_" + side] = sum(pr) / len(pr) if pr else None
            rec["proxy_kd_" + side] = sum(pk) / len(pk) if pk else None
            rec["proxy_n_" + side] = nk
            rec["stab_" + side] = stab.get(team, 0)
            rec["chg_" + side] = 1 if (team in last_lu and team in prev_lu
                                       and last_lu[team] != prev_lu[team]) else 0
            # 「真换人」：只有两场阵容都凑齐 5 个名字时才能判定，
            # 否则 3 人 vs 4 人也会被判成"换了"，把子集污染掉（老赛季缺名严重）
            rec["chg5_" + side] = 1 if (len(last_lu.get(team, set())) == 5
                                        and len(prev_lu.get(team, set())) == 5
                                        and last_lu[team] != prev_lu[team]) else 0
            # 「本场是否刚换人」：代理阵容 vs 再上一场阵容
            rec["chgnow_" + side] = 1 if (team in last_lu and team in prev_lu
                                          and last_lu[team] != prev_lu[team]) else 0

            # --- 阵容质量的变化量：代理阵容 vs 两场前的阵容 ---
            orr = []
            for p in prev_lu.get(team, set()):
                v = _player_at(player_hist, p, t, rating)
                if v is not None and v[0] is not None:
                    orr.append(v[0])
            rec["old_rate_" + side] = sum(orr) / len(orr) if orr else None

            # --- 队伍聚合特征（不经选手名，全赛季可用） ---
            for key in ("rk", "rd", "kd"):
                rec["tagg_%s_%s" % (key, side)] = _team_at(agg_hist, team, t, key)

            # --- 真实阵容特征（上限参考） ---
            ar, ak = [], []
            for p in actual:
                v = _player_at(player_hist, p, t, rating)
                if v is None:
                    continue
                if v[0] is not None:
                    ar.append(v[0])
                if v[1] is not None:
                    ak.append(v[1])
            rec["act_rate_" + side] = sum(ar) / len(ar) if ar else None
            rec["act_kd_" + side] = sum(ak) / len(ak) if ak else None

        rec["any_change"] = 1 if (rec["chg_A"] or rec["chg_B"]) else 0
        rec["real_change"] = 1 if (rec["chg5_A"] or rec["chg5_B"]) else 0
        rec["trust_both"] = 1 if (rec["trust_A"] and rec["trust_B"]) else 0
        # 双方代理阵容都凑齐 5 个名字（老赛季缺名严重，这会让特征变噪声）
        rec["full_both"] = 1 if (len(rec["proxy_A"]) == 5 and len(rec["proxy_B"]) == 5) else 0
        # ★ 代理阵容集：两队都能拿到「上一场可信阵容」，且各有足够的历史指标可算。
        #    这就是用户要的口径 —— 阵容名单只取自可信场次，但不要求本场本身可信。
        rec["proxy_ok"] = 1 if (rec["proxy_n_A"] >= MIN_NAMED
                                and rec["proxy_n_B"] >= MIN_NAMED) else 0
        out[mid] = rec

        # --- 更新状态（用本场真实数据，供之后的比赛用） ---
        # ⚠️ 只有**通过可信度闸门**的块才写进历史，否则名字是错的，会污染之后所有场次
        for team in (m["teamA"], m["teamB"]):
            ta = tagg.get((mid, team))
            if ta:
                agg_hist[team].append((t, ta))          # 聚合口径与名字无关，全量写入
            if not trusted[team]:
                continue
            for p in lineup.get((mid, team), set()):
                s = stat.get((mid, team, p))
                if s:
                    player_hist[p].append((t, s))
            actual = lineup.get((mid, team), set())
            if actual:
                if team in last_lu and last_lu[team] != actual:
                    stab[team] = 1
                elif team in last_lu and last_lu[team] == actual and len(actual) == 5:
                    stab[team] = stab.get(team, 1) + 1
                elif team not in last_lu:
                    stab[team] = 1
                prev_lu[team] = last_lu.get(team, set())
                last_lu[team] = actual
    return out


def _player_at(player_hist, p, t, mode="rk"):
    """选手在时刻 t 的衰减加权 (个人能力, KD)，只用 t 之前的记录。

    mode 决定「个人能力」用哪个口径：rk=每回合击杀 / rd=每回合伤害 / kd=击杀死亡比
    """
    h = [x for x in player_hist.get(p, []) if x[0] < t]
    if not h:
        return None
    r = weighted([(days_between(x[0], t) or MIN_DAYS, x[1].get(mode))
                  for x in h if x[1].get(mode) is not None], HALFLIFE)
    k = weighted([(days_between(x[0], t) or MIN_DAYS, x[1].get("kd"))
                  for x in h if x[1].get("kd") is not None], HALFLIFE)
    return (r, k)


def lu_vec(rec, which):
    """把阵容特征拼成向量。which 决定取哪一套。

    'proxy'  = 代理阵容（用上一场阵容）+ 稳定度 + 换人标志 + 质量变化
    'act'    = 真实阵容（本场真实阵容），只有水平项
    """
    def dif(a, b):
        if a is None or b is None:
            return 0.0
        return a - b

    if which == "act":
        return [dif(rec.get("act_rate_A"), rec.get("act_rate_B")),
                dif(rec.get("act_kd_A"), rec.get("act_kd_B"))]

    rate = dif(rec.get("proxy_rate_A"), rec.get("proxy_rate_B"))
    kd = dif(rec.get("proxy_kd_A"), rec.get("proxy_kd_B"))
    stab = min(rec.get("stab_A", 0), 10) - min(rec.get("stab_B", 0), 10)
    chg = rec.get("chg_A", 0) - rec.get("chg_B", 0)
    # 质量变化：代理阵容相对两场前阵容的提升
    da = (rec.get("proxy_rate_A") - rec.get("old_rate_A")
          if rec.get("proxy_rate_A") is not None and rec.get("old_rate_A") is not None else 0.0)
    db = (rec.get("proxy_rate_B") - rec.get("old_rate_B")
          if rec.get("proxy_rate_B") is not None and rec.get("old_rate_B") is not None else 0.0)
    # 队伍聚合（干净、全赛季可用）
    trk = dif(rec.get("tagg_rk_A"), rec.get("tagg_rk_B"))
    trd = dif(rec.get("tagg_rd_A"), rec.get("tagg_rd_B"))
    tkd = dif(rec.get("tagg_kd_A"), rec.get("tagg_kd_B"))
    return [rate, kd, float(stab), float(chg), da - db, trk, trd, tkd]


# 特征名 -> 在 lu_vec('proxy') 里的下标
LU_IDX = {k: i for i, k in enumerate(LU_DOC)}

# 预先定义好的几个变体（名字 -> 取哪几维）
VARIANTS = {
    "base": [],
    "all5": ["lu_rate_diff", "lu_kd_diff", "lu_stab_diff", "lu_chg_diff", "lu_delta_diff"],
    "level_only": ["lu_rate_diff", "lu_kd_diff"],          # 只是阵容水平（预期冗余）
    "change_only": ["lu_stab_diff", "lu_chg_diff", "lu_delta_diff"],   # 只保留「变动」信息
    "stab_only": ["lu_stab_diff"],
    "delta_only": ["lu_delta_diff"],
    "tagg_rk": ["tagg_rk_diff"],                            # 队伍聚合：每回合击杀
    "tagg_rd": ["tagg_rd_diff"],                            # 队伍聚合：每回合伤害
    "tagg_kd": ["tagg_kd_diff"],                            # 队伍聚合：KD
    "tagg_all": ["tagg_rk_diff", "tagg_rd_diff", "tagg_kd_diff"],
}


def make_builder(lf, variant):
    """返回 extra_of(matchid) -> list，供 run() 用。"""
    names = VARIANTS[variant]
    idx = [LU_IDX[k] for k in names]

    def extra_of(mid):
        if not names:
            return []
        r = lf.get(mid)
        if not r:
            return [0.0] * len(names)
        v = lu_vec(r, "proxy")
        return [v[i] for i in idx]
    return extra_of


# ------------------------------------------------------------------ 训练（与内核同超参）
def train_lr(X, y):
    n, d = len(X), len(X[0])
    mu = [sum(X[i][j] for i in range(n)) / n for j in range(d)]
    sd = []
    for j in range(d):
        v = sum((X[i][j] - mu[j]) ** 2 for i in range(n)) / n
        sd.append(math.sqrt(v) or 1.0)
    Z = [[(X[i][j] - mu[j]) / sd[j] for j in range(d)] for i in range(n)]
    w, b = [0.0] * d, 0.0
    for _ in range(EPOCHS):
        gw, gb = [0.0] * d, 0.0
        for i in range(n):
            zi = Z[i]
            z = b + sum(w[j] * zi[j] for j in range(d))
            e = sigmoid(z) - y[i]
            for j in range(d):
                gw[j] += e * zi[j]
            gb += e
        for j in range(d):
            w[j] -= LR * (gw[j] / n + L2 * w[j])
        b -= LR * (gb / n)
    return {"w": w, "b": b, "mu": mu, "sd": sd}


def predict(model, x):
    z = model["b"] + sum(model["w"][j] * ((x[j] - model["mu"][j]) / model["sd"][j])
                         for j in range(len(x)))
    return sigmoid(z)


# ------------------------------------------------------------------ 滚动前向
def run(mids, by_match, extra_of, eval_mapseq=0):
    """extra_of(matchid) -> 额外特征列表（无则返回 []）"""
    res = []
    cache, model = [], None
    for i, mid in enumerate(mids):
        if i >= WARMUP:
            if (i - WARMUP) % RETRAIN == 0:
                X, y = [], []
                for prev in cache:
                    for r in by_match[prev]:
                        if r["y_match"] is None:
                            continue
                        X.append(featurize(r) + extra_of(prev))
                        y.append(r["y_match"])
                model = train_lr(X, y)
            for r in by_match[mid]:
                if r["y_match"] is None:
                    continue
                if eval_mapseq is not None and r["mapseq"] != eval_mapseq:
                    continue
                p = predict(model, featurize(r) + extra_of(mid))
                res.append((mid, p, r["y_match"]))
        cache.append(mid)
    return res


def metrics(res):
    n = len(res)
    if not n:
        return (0.0, 0.0, 0.0, 0)
    acc = sum(1 for _, p, t in res if (p >= 0.5) == (t == 1)) / n
    ll = br = 0.0
    for _, p, t in res:
        p = min(max(p, 1e-6), 1 - 1e-6)
        ll += -(t * math.log(p) + (1 - t) * math.log(1 - p))
        br += (p - t) ** 2
    return acc, ll / n, br / n, n


def pair_test(a, b):
    """逐样本对数损失，a − b。返回 (平均差, 标准误, t)"""
    if len(a) != len(b) or not a:
        return None
    d = []
    for (_, pa, ta), (_, pb, tb) in zip(a, b):
        la = -(ta * math.log(min(max(pa, 1e-6), 1 - 1e-6))
               + (1 - ta) * math.log(1 - min(max(pa, 1e-6), 1 - 1e-6)))
        lb = -(tb * math.log(min(max(pb, 1e-6), 1 - 1e-6))
               + (1 - tb) * math.log(1 - min(max(pb, 1e-6), 1 - 1e-6)))
        d.append(la - lb)
    m = sum(d) / len(d)
    v = sum((x - m) ** 2 for x in d) / (len(d) - 1)
    se = math.sqrt(v / len(d))
    return m, se, (m / se if se else 0.0)


def subset(res, keep):
    """keep: set(matchid) —— 只保留这些场次的样本"""
    return [x for x in res if x[0] in keep]


# ------------------------------------------------------------------ 诊断
def diag(lf, mids, by_match, c):
    print("=" * 80)
    print("诊断一：「上一场阵容」作为代理，猜本场阵容猜得准吗")
    print("=" * 80)
    tot = exact = part = 0
    tot_t = exact_t = 0
    for mid in mids:
        r = lf.get(mid)
        if not r:
            continue
        for side in ("A", "B"):
            pr, ac = r.get("proxy_" + side) or set(), r.get("actual_" + side) or set()
            if len(pr) == 5 and len(ac) == 5:
                tot += 1
                if pr == ac:
                    exact += 1
                else:
                    part += 1
                if r.get("trust_both"):        # 只算「本场名字可信」的
                    tot_t += 1
                    if pr == ac:
                        exact_t += 1
    if tot:
        print(f"  可判定样本 {tot} 个「队-场次」")
        print(f"  代理阵容 == 本场阵容：{exact} 个 = {exact/tot*100:.1f}%")
        print(f"  代理阵容 ≠ 本场阵容：{part} 个 = {part/tot*100:.1f}%")
    if tot_t:
        print(f"  ⚠️ 只算通过可信度闸门的：{tot_t} 个，猜中 {exact_t} = "
              f"{exact_t/tot_t*100:.1f}%   ← 这个才是可信数字")

    # ★ 代理阵容集：本实验实际用的样本。规模 + 该集合内的代理准确率
    n_px = sum(1 for r in lf.values() if r.get("proxy_ok"))
    p_tot = p_exact = 0
    for mid in mids:
        r = lf.get(mid)
        if not r or not r.get("proxy_ok"):
            continue
        for side in ("A", "B"):
            if not r.get("trust_" + side):
                continue        # 本场名字不可信，就不知道真实阵容，没法判对错
            pr, ac = r.get("proxy_" + side) or set(), r.get("actual_" + side) or set()
            if pr and ac:
                p_tot += 1
                if pr == ac:
                    p_exact += 1
    print(f"\n  ★ 能用「上一场可信阵容」作代理的场次：{n_px} 场 "
          f"= {n_px/max(len(lf),1)*100:.1f}%（这是本实验的样本规模）")
    if p_tot:
        print(f"  ★ 该集合内、本场名字可信的部分：{p_tot} 个「队-场次」，"
              f"代理猜中 {p_exact} = {p_exact/p_tot*100:.1f}%")

    # ---- 对照：换个代理口径 —— 用「两队上次交手」的阵容 ----
    info = {}
    for r in c.execute("""SELECT matchid, teamA, teamB, match_time FROM match
                          WHERE has_detail=1 AND is_official=1 AND is_placeholder=0"""):
        info[r["matchid"]] = (r["teamA"], r["teamB"], r["match_time"])
    pair_list = defaultdict(list)
    for mid, (a, b, t) in info.items():
        pair_list[tuple(sorted((a, b)))].append((t, mid))
    for k in pair_list:
        pair_list[k].sort()
    prev_h2h = {}
    for k, lst in pair_list.items():
        for i in range(1, len(lst)):
            prev_h2h[lst[i][1]] = lst[i - 1][1]

    lu = defaultdict(set)
    for r in c.execute("SELECT matchid, team, player FROM player_map WHERE player IS NOT NULL"):
        lu[(r["matchid"], r["team"])].add(r["player"])

    h_tot = h_exact = 0
    gaps = []
    for mid, (a, b, t) in info.items():
        m0 = prev_h2h.get(mid)
        if not m0:
            continue
        gaps.append(days_between(info[m0][2], t) or 0.0)
        for team in (a, b):
            pr, ac = lu[(m0, team)], lu[(mid, team)]
            if len(pr) == 5 and len(ac) == 5:
                h_tot += 1
                if pr == ac:
                    h_exact += 1
    print("\n  对照：如果改用「两队上次交手」的阵容当代理")
    if h_tot:
        print(f"    可判定样本 {h_tot} 个「队-场次」，猜中 {h_exact} = {h_exact/h_tot*100:.1f}%"
              f"（两次交手平均相隔 {sum(gaps)/len(gaps):.0f} 天）")
        print(f"    → 比「同队上一场」的 {exact/tot*100:.1f}% 差，代理应当用同队上一场"
              if tot else "")

    rc = sum(1 for r in lf.values() if r.get("real_change"))
    print(f"\n  真实换人（前后两场阵容都凑齐 5 人且不同）：{rc} 场 = {rc/max(len(lf),1)*100:.1f}%")

    # ---- 可信度闸门体检 ----
    cname = {}
    for r in c.execute("SELECT matchid, contestname FROM match"):
        cname[r["matchid"]] = r["contestname"]
    stat = defaultdict(lambda: {"n": 0, "ok": 0, "cons": 0.0})
    for mid, rec in lf.items():
        s = stat[cname.get(mid, "?")]
        s["n"] += 1
        s["cons"] += (rec.get("cons_A", 0) + rec.get("cons_B", 0)) / 2
        if rec.get("trust_both"):
            s["ok"] += 1
    print("\n" + "=" * 84)
    print(f"诊断四：可信度闸门（块内具名选手里，在当赛季名单内的比例 ≥ {MIN_CONS:.0%}"
          f"，且至少 {MIN_NAMED} 个具名）")
    print("=" * 84)
    print(f'{"赛季":<16}{"场次":>7}{"平均一致率":>12}{"双队都通过":>12}{"通过率":>9}')
    print("-" * 84)
    tot = ok = 0
    for cn in sorted(stat):
        s = stat[cn]
        if s["n"] < 5:
            continue
        tot += s["n"]
        ok += s["ok"]
        print(f'{cn:<16}{s["n"]:>7}{s["cons"]/s["n"]*100:>11.1f}%'
              f'{s["ok"]:>12}{s["ok"]/s["n"]*100:>8.1f}%')
    print("-" * 84)
    print(f'合计 {tot} 场，双队都通过 {ok} 场 = {ok/max(tot,1)*100:.1f}%')
    print("→ 只有通过的场次才配谈「按选手名区分到人」；其余场次的选手特征一律置零。")

    print("\n" + "=" * 80)
    print("诊断二：新增特征的取值分布（判断有没有信息量）")
    print("=" * 80)
    vals = defaultdict(list)
    for mid in mids:
        r = lf.get(mid)
        if not r:
            continue
        v = lu_vec(r, "proxy")
        for k, x in zip(LU_DOC, v):
            vals[k].append(x)
    print(f'{"特征":<16}{"非零比例":>10}{"均值":>10}{"标准差":>10}{"最小":>9}{"最大":>9}')
    print("-" * 80)
    for k, xs in vals.items():
        nz = sum(1 for x in xs if abs(x) > 1e-9) / len(xs)
        m = sum(xs) / len(xs)
        sd = math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))
        print(f'{k:<16}{nz*100:>9.1f}%{m:>10.4f}{sd:>10.4f}{min(xs):>9.3f}{max(xs):>9.3f}')

    print("\n" + "=" * 80)
    print("诊断三：选手级特征单独看，跟胜负有没有相关（赛前点）")
    print("=" * 80)
    pre = []
    for mid in mids:
        r = lf.get(mid)
        if not r:
            continue
        for row in by_match[mid]:
            if row["mapseq"] == 0 and row["y_match"] is not None:
                pre.append((r, row["y_match"], row.get("kf_diff") or 0.0))
    for which, tag in (("proxy", "代理阵容"), ("act", "真实阵容")):
        v = [lu_vec(r, which)[0] for r, _, _ in pre]
        y = [t for _, t, _ in pre]
        kf = [k for _, _, k in pre]
        print(f"  {tag}「个人能力差」：与胜负相关 {_corr(v, y):+.4f}，"
              f"与卡尔曼实力差相关 {_corr(v, kf):+.4f}")
    print(f"  对照：卡尔曼实力差与胜负相关 {_corr([k for _, _, k in pre], [t for _, t, _ in pre]):+.4f}")


def _corr(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    num = sum((x[i] - mx) * (y[i] - my) for i in range(n))
    dx = math.sqrt(sum((a - mx) ** 2 for a in x))
    dy = math.sqrt(sum((b - my) ** 2 for b in y))
    return num / (dx * dy) if dx and dy else 0.0


# ------------------------------------------------------------------ 主流程
def main():
    global MIN_CONS, MIN_NAMED
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag", action="store_true", help="只看诊断")
    ap.add_argument("--rating", default="rk", choices=["rk", "rd", "kd"],
                    help="个人能力口径：rk=每回合击杀 / rd=每回合伤害 / kd=击杀死亡比")
    ap.add_argument("--min-cons", type=float, default=MIN_CONS,
                    help="可信度闸门门槛：块内具名选手在当赛季名单内的最低比例（默认 0.6）")
    ap.add_argument("--min-named", type=int, default=MIN_NAMED,
                    help="块内至少要有几个具名选手才够判断（默认 3）")
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS),
                    help="只跑指定的特征组合")
    ap.add_argument("--set", dest="which_set", default="all",
                    choices=["all", "trusted", "proxy"],
                    help="评估样本集：all=全部（特征缺则置零）/ "
                         "trusted=只用本场两个队块都可信的 / "
                         "proxy=用「上一场可信阵容」作代理、只要能算出来就纳入")
    ap.add_argument("--trusted-only", action="store_true",
                    help="（旧写法）等价于 --set trusted")
    args = ap.parse_args()

    MIN_CONS, MIN_NAMED = args.min_cons, args.min_named

    c = connect()
    mids, by_match = load_points(c)
    print(f"载入 {len(mids)} 场比赛、{sum(len(v) for v in by_match.values())} 个预测点")
    lf = build_lineup_feats(c, rating=args.rating)
    print(f"阵容特征覆盖 {len(lf)} 场　个人能力口径 = {args.rating}\n")

    which = "trusted" if args.trusted_only else args.which_set
    if which != "all":
        key = "trust_both" if which == "trusted" else "proxy_ok"
        keep = {mid for mid, r in lf.items() if r.get(key)}
        names = {"trusted": "本场两个队块都通过可信度闸门",
                 "proxy": "能用「上一场可信阵容」作代理"}[which]
        mids = [m for m in mids if m in keep]
        by_match = {k: v for k, v in by_match.items() if k in keep}
        print(f"★ 评估集 =「{names}」：{len(mids)} 场\n")

    if args.diag:
        diag(lf, mids, by_match, c)
        return

    print("=" * 88)
    print("滚动前向对比（评估「开赛前」这个时点，口径与 backtest.py 一致）")
    print("=" * 88)
    runs = {}
    order = args.variants
    for v in order:
        tag = v if v != "base" else "base（基线 15 维）"
        res = run(mids, by_match, make_builder(lf, v), eval_mapseq=0)
        runs[v] = res
        acc, ll, br, n = metrics(res)
        print(f'{tag:<28}{acc*100:>7.1f}%   对数损失 {ll:.4f}   布里尔 {br:.4f}   样本 {n}')

    # 真实阵容（上限参考）。⚠️ 只有本场两个队块都通过闸门，「真实阵容」才真的是真实阵容，
    # 否则那串名字本身是错的 —— 一律置零，不给模型喂假信息。
    def ex_oracle(mid):
        r = lf.get(mid)
        if not r or not (r.get("trust_A") and r.get("trust_B")):
            return [0.0, 0.0]
        return lu_vec(r, "act")

    res = run(mids, by_match, ex_oracle, eval_mapseq=0)
    runs["oracle"] = res
    acc, ll, br, n = metrics(res)
    print(f'{"oracle（真实阵容，上限）":<28}{acc*100:>7.1f}%   对数损失 {ll:.4f}   布里尔 {br:.4f}   样本 {n}')

    print("-" * 88)
    if "base" in runs:
        base = runs["base"]
        for v in order + ["oracle"]:
            if v == "base":
                continue
            pt = pair_test(base, runs[v])
            if pt:
                m, se, t = pt
                verdict = "显著（|t|>2）" if abs(t) > 2 else "不显著，可能只是噪声"
                print(f'  {v:<22} vs 基线：对数损失差 {m:+.4f}　标准误 {se:.4f}　'
                      f't = {t:+.2f}　→ {verdict}')

    # ---- 针对性检验：整体效应会被「没换人 / 缺名」的场次平均掉，所以切开看 ----
    if "base" in runs:
        subsets = [
            ("有一队刚换人（含缺名误判）",
             {mid for mid, r in lf.items() if r.get("any_change")}),
            ("发生真实换人（两场都凑齐 5 人才判定）",
             {mid for mid, r in lf.items() if r.get("real_change")}),
            ("双方代理阵容名字齐全（5/5）",
             {mid for mid, r in lf.items() if r.get("full_both")}),
            ("★ 双方队块都通过可信度闸门",
             {mid for mid, r in lf.items() if r.get("trust_both")}),
        ]
        for tag, keep in subsets:
            print("-" * 88)
            print(f'【只在「{tag}」的场次上比】共 {len(keep)} 场')
            for v in order:
                if v == "base":
                    continue
                a = subset(runs["base"], keep)
                b = subset(runs[v], keep)
                if len(a) < 20:
                    continue
                acc_a, ll_a, _, _ = metrics(a)
                acc_b, ll_b, _, _ = metrics(b)
                pt = pair_test(a, b)
                tt = f"t = {pt[2]:+.2f}" if pt else ""
                print(f'  {v:<20} 基线 {acc_a*100:>5.1f}% / {ll_a:.4f}　→　'
                      f'{acc_b*100:>5.1f}% / {ll_b:.4f}　{tt}')

    print("=" * 88)
    print("对数损失/布里尔越小越好；命中率越大越好。")
    print("配对检验算的是「基线 − 变体」：差值为正 = 变体更好，为负 = 变体更差。")


if __name__ == "__main__":
    main()
