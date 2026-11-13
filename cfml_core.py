# -*- coding: utf-8 -*-
"""cfml_core.py —— CFML 预测内核（「局前预测」与「实时预测」两个模式共用）

这一层只负责「算概率」：不碰命令行、不碰界面、不碰截屏。
两个模式都 import 它，保证两边口径完全一致、不会各算各的。

    局前模式  predict.py prematch   输入两队队名（可选开赛时间）
    实时模式  predict.py live       输入当前大比分 + 当前局回合比分

## 三条铁律（改代码前先读）

1. **赛前特征必须按时间截断**：算某场比赛的赛前特征时，只用「那场开赛之前」的比赛。
   所以同一套代码既能滚动回测，也能预测未来未开赛的比赛，还能预测正在打的比赛
   （拿它的开赛时间当截断点）。顺序一旦错，回测分数就是假的。

2. **只用逻辑回归 + 15 个特征**。样本量是 2000 多个预测点，落在 12 赛季 × 11 队里，
   有效自由度远小于样本数，上复杂模型只会过拟合。别换。

3. **训练只能用「预测点之前」的比赛**，不能随机划分（同一场比赛有多个预测点，会泄漏）。

## 对外主要接口

    conn = connect()
    ms, maps_of, deaths = load_all(conn)
    hist, h2h, szn = slice_history(ms, maps_of, asof="2026-09-26 19:30:00")
    pre = prematch_feats(hist, h2h, szn, "XROCK.快手", "EP.快手", asof, cid)
    x   = featurize(make_row(pre, live=None))          # 15 维
    model = train_logreg(X, y)
    p   = sigmoid_predict(model, x)

    rounds_over = fit_round_market(conn)               # 单局回合数盘口
    lead_model  = fit_lead_model(conn)                 # 局中领先 -> 本局胜率
"""
import math
import os
import re
import sqlite3
from collections import defaultdict
from datetime import datetime

from build_db import norm_team  # 队名归一化，共用一份，别在这里另写一套

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "cfml.db")

HALFLIFE = 365.0     # 时间衰减半衰期（天）
MIN_DAYS = 1.0

# ---------------------------------------------------------------- 模型特征
# 顺序不能改！训练和预测必须严格同序。
FEATS = [
    "wr_diff", "mapwr_diff", "netmaps_diff", "rd_diff", "exp_diff",
    "sznwr_diff", "h2h_wr", "h2h_last",
    "score_diff", "rnd_diff", "kd_diff", "last_map_won", "maps_played",
    "is_playoff", "score_x_maps",
]

FEAT_DOC = {
    "wr_diff": "近期衰减胜率差（A-B）",
    "mapwr_diff": "近期单局胜率差",
    "netmaps_diff": "近期平均净胜局差",
    "rd_diff": "近期平均回合分差 ÷10",
    "exp_diff": "历史经验量差（衰减后场次的对数差）",
    "sznwr_diff": "本赛季胜率差",
    "h2h_wr": "历史交手胜率（A 视角，居中）",
    "h2h_last": "上次交手 A 是否赢（居中）",
    "score_diff": "当前大比分差（A-B）",
    "rnd_diff": "已打局回合总分差 ÷10",
    "kd_diff": "已打局队伍 KD 差",
    "last_map_won": "上一局 A 是否赢（居中）",
    "maps_played": "已打完几局",
    "is_playoff": "是否季后赛",
    "score_x_maps": "交互项：领先 × 已打局数",
}


# ---------------------------------------------------------------- 阶段口径
# 「是不是季后赛」这个判断必须在**训练和预测两处用同一份实现**，否则特征会静默错位。
#
# 历史踩坑（2026-09-29 修）：build_features.py 用的是 ("季后赛", "决赛")、
# predict.py 只查 "季后赛"。结果是：
#   「胜者组决赛」→ 训练标 1、预测标 0（不一致）
#   「胜者组第一轮」「败者组第一轮」→ 两边都漏标成 0（2022~2025 的季后赛全漏了）
PLAYOFF_KEYS = ("季后赛", "决赛", "胜者组", "败者组", "淘汰赛")


def is_playoff_of(subcontestname):
    """轮次名 -> 是否季后赛（淘汰赛阶段）。训练与预测必须都调这个。

    常规赛的轮次名一律含"常规赛"，先按它排除；其余命中任一淘汰赛关键词就算季后赛。
    """
    s = subcontestname or ""
    if "常规赛" in s:
        return 0
    return 1 if any(k in s for k in PLAYOFF_KEYS) else 0


def match_format(subcontestname, contestname=None):
    """轮次名（+赛季名）-> 赛制（BO 几，即先赢几局）。训练/预测/抓取必须都调这个。

    实测规律（900 场，2026-09-30 修正）：
      常规赛                            → BO3（所有赛季）
      2023 起的季后赛（胜者组/败者组等）→ BO5
      2021 / 2022 的季后赛              → **BO7**（先赢 4 局）！
        12 场实测比分 4:3 / 4:2（2022 夏 10 场 + 2022 秋的胜者组/败者组 5 场），
        当年「季后赛」「胜者组第一轮」这些名字全是 BO7，2023 起才改 BO5。
      总决赛 / 决赛                     → BO7（所有赛季）
      半决赛                            → BO5（⚠️ 要先于"决赛"判断，"半决赛"含"决赛"二字）
    ⚠️ 千万别拿「这场实际打了几局」(n_maps) 当赛制——BO5 横扫 3:0 时 n_maps=3，
    会被误判成 BO3，比分分布整个算错；未开赛时 n_maps 干脆是 0。
    """
    s = (subcontestname or "")
    c = (contestname or "")
    # 2021/2022 赛季名里带年份（如 "2022CFML夏季赛"）。这两年的淘汰赛全是 BO7。
    old_bo7 = ("2021" in c or "2022" in c)
    if "常规赛" in s:
        return 3
    if "胜者组" in s or "败者组" in s:
        return 7 if old_bo7 else 5
    if "半决赛" in s or "季军赛" in s:    # 必须在"决赛"之前判（"半决赛"含"决赛"）
        return 7 if old_bo7 else 5
    if "决赛" in s:                      # 含"总决赛"
        return 7
    if "季后赛" in s:
        return 7 if old_bo7 else 5
    if "季后赛" in c:                    # 兜底：轮次名认不出来，按赛季类型猜
        return 7 if old_bo7 else 5
    return 3


def maps_needed(best_of):
    """BO 几 -> 打完需要几局（先赢 ceil(bo/2) 局）。BO3→2、BO5→3、BO7→4。"""
    return best_of // 2 + 1


def connect():
    if not os.path.exists(DB):
        raise SystemExit(f"找不到数据库 {DB}，先跑 python build_db.py")
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    return c


# ---------------------------------------------------------------- 地图名归一化
# 同一张图在库里有多套名字：重制版加"新"，高清版加"-高清"，结构优化加"（结构优化）"。
# 不归一的话统计会碎成十几份，每份样本都不够。
MAP_VARIANTS = ("（结构优化）", "(结构优化)", "-高清", "－高清")
MAP_PREFIXES = ("新", "夏日")


def norm_map(name):
    """地图名 -> 基础图名。'新潜艇' / '潜艇' 都归成 '潜艇'。

    只做版本差异的归并，不做同义合并（例如"离岛"和"沙丘"是两张图，保持分开）。
    原始名保留在 `map.map_name` 列里，想按原版统计随时可以取。
    """
    s = (name or "").strip()
    if not s:
        return None
    for v in MAP_VARIANTS:
        s = s.replace(v, "")
    for p in MAP_PREFIXES:
        if s.startswith(p) and len(s) > len(p):
            s = s[len(p):]
            break
    return s.strip() or None


# ---------------------------------------------------------------- 工具
def parse_time(t):
    try:
        return datetime.strptime(str(t)[:19], "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def days_between(t1, t2):
    a, b = parse_time(t1), parse_time(t2)
    if a is None or b is None:
        return None
    return (b - a).total_seconds() / 86400.0


def weighted(pairs, halflife):
    """pairs = [(天数差, 值)]，衰减加权平均；halflife<=0 退化为简单平均"""
    if not pairs:
        return None
    if not halflife or halflife <= 0:
        return sum(v for _, v in pairs) / len(pairs)
    num = den = 0.0
    for age, v in pairs:
        w = 0.5 ** (max(age, 0.0) / halflife)
        num += w * v
        den += w
    return num / den if den else None


def sigmoid(z):
    z = max(-30.0, min(30.0, z))
    return 1.0 / (1.0 + math.exp(-z))


# ---------------------------------------------------------------- 装载
def load_all(c):
    """把全部正式比赛（含逐局）读进内存。约 900 场，很轻。"""
    ms = [dict(r) for r in c.execute("""
        SELECT matchid, contestid, contestname, subcontestname, match_time,
               teamA, teamB, n_maps, winner
        FROM match
        WHERE is_official=1 AND is_placeholder=0 AND has_detail=1
          AND teamA IS NOT NULL AND teamB IS NOT NULL AND match_time IS NOT NULL
        ORDER BY match_time, matchid
    """)]
    maps_of = defaultdict(list)
    for r in c.execute("""
        SELECT matchid, map_seq, teamA, scoreA, scoreB, winner FROM map
        ORDER BY matchid, map_seq
    """):
        maps_of[r["matchid"]].append(dict(r))
    # 选手死亡数（算队伍 KD 用）
    deaths = defaultdict(int)
    for r in c.execute("""
        SELECT matchid, map_seq, team, SUM(death) d FROM player_map
        WHERE team IS NOT NULL GROUP BY matchid, map_seq, team
    """):
        deaths[(r["matchid"], r["map_seq"], r["team"])] = r["d"] or 0
    return ms, maps_of, deaths


# ---------------------------------------------------------------- 历史切片
def slice_history(ms, maps_of, asof=None):
    """把「asof 之前」的比赛整理成历史，供算赛前特征。

    返回 (hist, h2h, season_games)
      hist[team]        = [{'time','won','mw','ml','nm','rf','ra'}]
      h2h[(t1,t2)]      = [{'time','a_won'}]   a_won 是元组里第一个队是否赢
      season_games[(team,contestid)] = [(time, won)]
    """
    cutoff = parse_time(asof) if asof else None
    hist = defaultdict(list)
    h2h = defaultdict(list)
    season_games = defaultdict(list)

    for m in ms:
        t = m["match_time"]
        if cutoff is not None:
            pt = parse_time(t)
            if pt is None or pt >= cutoff:
                continue
        a, b = m["teamA"], m["teamB"]
        if a == b:
            continue
        maps = maps_of.get(m["matchid"]) or []
        if not maps:
            continue
        a_won = (m["winner"] == a)
        a_mw = sum(1 for g in maps if g["winner"] == a)
        b_mw = sum(1 for g in maps if g["winner"] == b)
        a_rf = sum((g["scoreA"] or 0) for g in maps)
        a_ra = sum((g["scoreB"] or 0) for g in maps)
        nm = len(maps)

        hist[a].append(dict(time=t, won=1 if a_won else 0, mw=a_mw, ml=b_mw,
                            nm=nm, rf=a_rf, ra=a_ra))
        hist[b].append(dict(time=t, won=0 if a_won else 1, mw=b_mw, ml=a_mw,
                            nm=nm, rf=a_ra, ra=a_rf))
        season_games[(a, m["contestid"])].append((t, 1 if a_won else 0))
        season_games[(b, m["contestid"])].append((t, 0 if a_won else 1))
        h2h[tuple(sorted((a, b)))].append({"time": t, "a_won": a_won})
    return hist, h2h, season_games


def prematch_feats(hist, h2h, season_games, a, b, asof, cid, halflife=HALFLIFE):
    """算 a vs b 在 asof 这一刻的赛前特征（只用 asof 之前的比赛）"""
    def team_stats(team):
        past = [x for x in hist.get(team, []) if x["time"] < asof]
        if not past:
            return dict(wr=None, mapwr=None, netmaps=None, rd=None, n=0.0)
        age = lambda x: max(days_between(x["time"], asof) or 0.0, MIN_DAYS)
        return dict(
            wr=weighted([(age(x), x["won"]) for x in past], halflife),
            mapwr=weighted([(age(x), x["mw"] / x["nm"] if x["nm"] else 0.5) for x in past], halflife),
            netmaps=weighted([(age(x), x["mw"] - x["ml"]) for x in past], halflife),
            rd=weighted([(age(x), x["rf"] - x["ra"]) for x in past], halflife),
            n=(sum(0.5 ** (age(x) / halflife) for x in past) if halflife and halflife > 0
               else float(len(past))),
        )

    fa, fb = team_stats(a), team_stats(b)

    def szn(team):
        past = [x for x in season_games.get((team, cid), []) if x[0] < asof]
        return (sum(v for _, v in past) / len(past)) if past else None

    key = tuple(sorted((a, b)))
    pair = [x for x in h2h.get(key, []) if x["time"] < asof]
    if pair:
        a_is_first = (key[0] == a)
        wins_a = sum(1 for x in pair if (x["a_won"] if a_is_first else not x["a_won"]))
        h2h_wr, h2h_n = wins_a / len(pair), len(pair)
        last = pair[-1]
        h2h_last = 1 if (last["a_won"] if a_is_first else not last["a_won"]) else 0
    else:
        h2h_wr, h2h_n, h2h_last = None, 0, None

    return dict(a_wr=fa["wr"], b_wr=fb["wr"], a_mapwr=fa["mapwr"], b_mapwr=fb["mapwr"],
                a_netmaps=fa["netmaps"], b_netmaps=fb["netmaps"],
                a_rd=fa["rd"], b_rd=fb["rd"], a_n=fa["n"], b_n=fb["n"],
                a_sznwr=szn(a), b_sznwr=szn(b),
                h2h_wr=h2h_wr, h2h_n=h2h_n, h2h_last=h2h_last)


# ---------------------------------------------------------------- 卡尔曼实力跟踪
# 替掉原来的「手工半衰期」。
# 实测依据见 exp_kalman.py：对数损失 0.5580 → 0.5275（改善 5.5%），配对检验 t = +3.49 显著。
#
# 状态 = 每队实力 x（logit 尺度），观测 = 二值胜负，观测方程 P(A赢) = sigmoid(x_A − x_B)。
# 观测是非线性的，所以严格说是**扩展卡尔曼滤波（EKF）**：对 sigmoid 做一阶泰勒展开。
# 每个队各带一个协方差：协方差大 = 信息少 = 该队更新快 —— 这就是"每队自适应的学习率"，
# 正好对应「大换血的队更新快、没换人的队更新慢」。
KF_Q = 0.002         # 过程噪声率（每天给协方差加多少），等价于"半衰期"
KF_R = 0.25          # 观测噪声（二值观测的伯努利方差）
KF_SIGMA0 = 1.0      # 新队初始实力的标准差


def kalman_strength(ms, asof=None, q=KF_Q, r=KF_R, sigma0=KF_SIGMA0):
    """跑扩展卡尔曼滤波跟踪队伍实力。

    返回 (out, state)：
      out   = {matchid: (x_a, x_b, P_aa, P_bb)} —— 每场**开赛前**的估计，
              用来给 feature_point 表填特征（每场都是"先记状态、再更新"，天然无泄漏）
      state = {'teams', 'idx', 'x', 'P'} —— 跑完后的**最终状态**，
              用来预测「还没开赛、库里查不到」的比赛

    asof 给了就只跑 asof 之前的比赛（红线①：赛前特征必须按时间截断）。
    ms 必须是按时间升序的比赛列表（load_all 已经排好序了）。
    """
    teams = sorted({m["teamA"] for m in ms} | {m["teamB"] for m in ms})
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    x = [0.0] * n
    P = [[(sigma0 ** 2 if i == j else 0.0) for j in range(n)] for i in range(n)]

    out = {}
    last = None
    for m in ms:
        if asof is not None and m["match_time"] >= asof:
            continue                      # 时间截断：asof 之后的比赛一律不看
        t = parse_time(m["match_time"])
        # 时间流逝 -> 不确定性增长（这就是"遗忘"，替代了原来的固定半衰期）
        if last is not None and t is not None:
            dt = (t - last).total_seconds() / 86400.0
            if dt > 0:
                add = q * dt
                for i in range(n):
                    P[i][i] += add
        if t is not None:
            last = t

        a, b = idx[m["teamA"]], idx[m["teamB"]]
        out[m["matchid"]] = (x[a], x[b], P[a][a], P[b][b])   # 先记录，再更新

        w = m["winner"]
        if w not in (m["teamA"], m["teamB"]):
            continue
        p = sigmoid(x[a] - x[b])
        Ha = p * (1 - p)
        Hb = -Ha
        S = (Ha * Ha * P[a][a] + Ha * Hb * P[a][b]
             + Hb * Ha * P[b][a] + Hb * Hb * P[b][b] + r)
        if S <= 0:
            continue
        K = [(P[i][a] * Ha + P[i][b] * Hb) / S for i in range(n)]
        inn = (1.0 if w == m["teamA"] else 0.0) - p
        for i in range(n):
            x[i] += K[i] * inn
        HP = [Ha * P[a][j] + Hb * P[b][j] for j in range(n)]
        for i in range(n):
            ki = K[i]
            if ki == 0.0:
                continue
            Pi = P[i]
            for j in range(n):
                Pi[j] -= ki * HP[j]

    return out, {"teams": teams, "idx": idx, "x": x, "P": P}


def _conf_gap(pa, pb):
    """不确定性差：用 log 精度（−log P）。协方差小 = 信息足 = 确定。"""
    return math.log(max(pb, 1e-9)) - math.log(max(pa, 1e-9))


def kalman_feats(kf, matchid):
    """从 out 里取这场开赛前的 (实力差, 不确定性差)。

    返回 (None, None) 表示这场没查到 —— 调用方会回退到手工特征。
    """
    if not kf:
        return None, None
    st = kf.get(matchid)
    if st is None:
        return None, None
    xa, xb, pa, pb = st
    return xa - xb, _conf_gap(pa, pb)


def kalman_state_feats(state, ta, tb):
    """从最终状态里取两队的 (实力差, 不确定性差) —— 预测还没开赛的比赛时用。"""
    if not state:
        return None, None
    idx, x, P = state["idx"], state["x"], state["P"]
    i, j = idx.get(ta), idx.get(tb)
    if i is None or j is None:
        return None, None
    return x[i] - x[j], _conf_gap(P[i][i], P[j][j])


# ---------------------------------------------------------------- 拼一行特征
def make_row(pre, live=None, mapseq=0, is_playoff=0):
    """把「赛前特征 + 局中状态」拼成一行（字段名与 feature_point 表一致）"""
    row = dict(pre)
    row.setdefault("a_sznwr", None)
    row.setdefault("b_sznwr", None)
    row.update(dict(mapseq=mapseq, is_playoff=is_playoff,
                    score_diff=0, rnd_diff=0, a_kd=None, b_kd=None,
                    last_map_won=None))
    if live:
        row.update(live)
    return row


def featurize(r):
    """一行 -> 15 维模型输入。必须与训练时严格一致。"""
    def g(k, d=0.0):
        v = r.get(k)
        return d if v is None else v

    score_diff = g("score_diff")
    maps_played = g("mapseq")
    # 第 0 维（实力差）与第 4 维（经验量）：有卡尔曼特征就用卡尔曼的，
    # 没有就回退到原来的手工半衰期口径。这样新旧数据可以共存，迁移期不会炸。
    kfd, kfc = r.get("kf_diff"), r.get("kf_conf")
    wr_feat = kfd if kfd is not None else (g("a_wr", 0.5) - g("b_wr", 0.5))
    exp_feat = (kfc if kfc is not None
                else (math.log1p(max(g("a_n"), 0)) - math.log1p(max(g("b_n"), 0))))
    return [
        wr_feat,
        g("a_mapwr", 0.5) - g("b_mapwr", 0.5),
        g("a_netmaps") - g("b_netmaps"),
        (g("a_rd") - g("b_rd")) / 10.0,
        exp_feat,
        g("a_sznwr", 0.5) - g("b_sznwr", 0.5),
        g("h2h_wr", 0.5) - 0.5,
        g("h2h_last", 0.5) - 0.5,
        score_diff,
        g("rnd_diff") / 10.0,
        g("a_kd", 1.0) - g("b_kd", 1.0),
        g("last_map_won", 0.5) - 0.5,
        maps_played,
        g("is_playoff"),
        score_diff * maps_played,
    ]


# ---------------------------------------------------------------- 模型
EPOCHS = 260
LR = 0.5
L2 = 0.002


def train_logreg(X, y, epochs=EPOCHS, lr=LR, l2=L2):
    n = len(X)
    # ⚠️ 维数必须取自数据本身。2026-09-30 踩坑：曾写死 d = len(FEATS)，
    # 实验脚本往特征向量**追加**维度时，多出来的特征会被静默丢弃、实验全程无效
    # （势头特征实验因此测出"完全没反应"的假结论）。
    d = len(X[0]) if X else 0
    if n == 0 or d == 0:
        return None
    if any(len(row) != d for row in X):
        raise ValueError(f"特征向量长度不一致：期望 {d} 维，有 {sum(1 for r in X if len(r) != d)} 行不是")
    mu = [sum(X[i][j] for i in range(n)) / n for j in range(d)]
    sd = []
    for j in range(d):
        v = sum((X[i][j] - mu[j]) ** 2 for i in range(n)) / n
        sd.append(math.sqrt(v) or 1.0)
    Z = [[(X[i][j] - mu[j]) / sd[j] for j in range(d)] for i in range(n)]

    w, b = [0.0] * d, 0.0
    for _ in range(epochs):
        gw = [0.0] * d
        gb = 0.0
        for i in range(n):
            zi = Z[i]
            z = b
            for j in range(d):
                z += w[j] * zi[j]
            p = sigmoid(z)
            e = p - y[i]
            for j in range(d):
                gw[j] += e * zi[j]
            gb += e
        for j in range(d):
            w[j] -= lr * (gw[j] / n + l2 * w[j])
        b -= lr * (gb / n)
    return {"w": w, "b": b, "mu": mu, "sd": sd, "n": n}


def sigmoid_predict(model, x):
    if model is None:
        return 0.5
    z = model["b"]
    # 维数跟模型走，别写死 len(FEATS)（与 train_logreg 同一个坑）
    for j in range(len(model["w"])):
        z += model["w"][j] * ((x[j] - model["mu"][j]) / model["sd"][j])
    return sigmoid(z)


def train_on_feature_points(c, asof=None, mapseq=None):
    """用 asof 之前的 feature_point 训练整场胜负模型。

    mapseq=None 表示用全部预测点（与回测口径一致，实测 73.7% / 赛前 72.0%）；
    传具体数字则只用该时机的点训练。
    """
    sql = """SELECT * FROM feature_point
             WHERE y_match IS NOT NULL AND match_time IS NOT NULL"""
    args = []
    if asof:
        sql += " AND match_time < ?"
        args.append(asof)
    if mapseq is not None:
        sql += " AND mapseq = ?"
        args.append(mapseq)
    rows = [dict(r) for r in c.execute(sql, args)]
    X = [featurize(r) for r in rows]
    y = [r["y_match"] for r in rows]
    m = train_logreg(X, y)
    if m:
        m["n_maps_points"] = len(rows)
    return m, len(rows)


# ---------------------------------------------------------------- 盘口模型
def fit_round_market(c, epochs=900, lr=0.6):
    """单局回合数盘口：P(单局总回合 > 16.5) 随「两队实力差」怎么变。

    特征是 |赛前衰减胜率差|。只有 1 个特征、2300 个样本，几毫秒就能拟合完。
    """
    rows = c.execute("""
        SELECT ABS(fp.a_wr - fp.b_wr) d, (m.scoreA + m.scoreB) tot
        FROM map m
        JOIN match mt ON mt.matchid = m.matchid
        JOIN feature_point fp ON fp.matchid = m.matchid AND fp.mapseq = 0
        WHERE m.scoreA IS NOT NULL AND m.scoreB IS NOT NULL
          AND fp.a_wr IS NOT NULL AND fp.b_wr IS NOT NULL
          AND mt.is_official=1 AND mt.is_placeholder=0
    """).fetchall()
    if len(rows) < 100:
        return None
    X = [[r["d"]] for r in rows]
    y = [1.0 if r["tot"] > 16.5 else 0.0 for r in rows]
    mu = [sum(x[0] for x in X) / len(X)]
    sd = [math.sqrt(sum((x[0] - mu[0]) ** 2 for x in X) / len(X)) or 1.0]
    w, b = [0.0], 0.0
    for _ in range(epochs):
        gw, gb = 0.0, 0.0
        for i, x in enumerate(X):
            z = b + w[0] * ((x[0] - mu[0]) / sd[0])
            gw += (sigmoid(z) - y[i]) * ((x[0] - mu[0]) / sd[0])
            gb += (sigmoid(z) - y[i])
        w[0] -= lr * (gw / len(X))
        b -= lr * (gb / len(X))
    return {"w": w, "b": b, "mu": mu, "sd": sd, "n": len(rows)}


def pred_round_market(model, abs_wr_diff):
    """给定两队赛前胜率差（绝对值），返回 P(单局回合数 > 16.5)"""
    if model is None:
        return 0.49
    z = model["b"] + model["w"][0] * ((abs_wr_diff - model["mu"][0]) / model["sd"][0])
    return sigmoid(z)


def fit_lead_model(c, k=30.0):
    """局中领先模型：P(A 赢下本局 | 第 r 回合结束时领先 d 分)

    为什么不用一条逻辑曲线拟合：实测单条曲线抓不住非线性 ——
    「第 8 回合领先 2 分」只是略占优，而「第 16 回合领先 2 分」几乎定局；
    而且领先分很大时概率饱和得很快。一条线性（在 d 上）的曲线两头都抓不准。

    改成**分层平滑查表**：
      · 某个格子 (回合 r, 领先 d) 样本够 → 直接信数据
      · 样本少 → 向「只按领先分汇总」的先验收缩（相当于加 30 个虚拟样本）
    实测这套跟经验的偏差基本都在 1~2 个百分点内。
    """
    cells = {}
    for r in c.execute("""
        SELECT round_no r, cur_a - cur_b d, COUNT(*) n, SUM(a_wins_map) w
        FROM round_snap GROUP BY 1, 2
    """):
        cells[(r["r"], r["d"])] = (r["n"], r["w"] or 0)
    prior = {}
    for r in c.execute("""
        SELECT cur_a - cur_b d, COUNT(*) n, SUM(a_wins_map) w
        FROM round_snap GROUP BY 1
    """):
        prior[r["d"]] = (r["w"] or 0) / r["n"]
    if not cells:
        return None
    return {"cells": cells, "prior": prior, "k": k,
            "n_cells": len(cells), "n_snap": sum(v[0] for v in cells.values())}


def pred_map_from_lead(model, round_no, lead):
    """第 round_no 回合结束时领先 lead 分 -> 本局最终获胜概率（A 方视角）"""
    if not model:
        return 0.5
    d = max(-12, min(12, int(round(lead))))
    r = max(1, min(16, int(round_no)))
    prior = model["prior"]
    if d in prior:
        p0 = prior[d]
    else:                                   # 领先分超出常见范围，取最接近的那档
        p0 = prior[min(prior, key=lambda x: abs(x - d))]
    n, w = model["cells"].get((r, d), (0, 0))
    k = model["k"]
    return (w + k * p0) / (n + k)


# ------------------------------------------------- 市场共识对照（可选）
def odds_to_prob(odds_a, odds_b):
    """把两个赔率换成「去水后」的隐含概率。

    庄家两边赔率加起来倒数 > 1，多出来的那部分叫水位（抽水）。
    去掉水位才是市场真实看法。
    """
    if not odds_a or not odds_b or odds_a <= 1 or odds_b <= 1:
        return None
    ra, rb = 1.0 / odds_a, 1.0 / odds_b
    s = ra + rb
    return {"implied_A": ra / s, "implied_B": rb / s, "vig": s - 1.0}


def ev_of_bet(p_model, odds):
    """给定赔率下的期望收益：用来量化模型与市场共识的分歧有多大"""
    if not odds or odds <= 1:
        return None
    return p_model * odds - 1.0


# ---------------------------------------------------------------- 比分竞猜赔率
def parse_score_odds(text):
    """把命令行写的比分赔率解析成 dict。

    形如 "2:0=6.20,2:1=3.81,1:2=2.34,0:2=2.07"。
    ⚠️ 半角/全角标点都要认（`，` `、` `；` `＝` `：`）—— 从中文界面复制过来的
    常常是全角，不认就会静默少解析几项。解析不出来的一律忽略，返回 {} 表示没给。
    """
    s = (text or "").strip()
    for a, b in (("＝", "="), ("：", ":"), ("，", ","), ("、", ","), ("；", ";")):
        s = s.replace(a, b)
    out = {}
    for part in re.split(r"[,;\s]+", s):
        if not part or "=" not in part:
            continue
        k, _, v = part.partition("=")
        k = k.strip()
        try:
            v = float(v)
        except ValueError:
            continue
        if v > 1:
            out[k] = v
    return out


def score_odds_to_prob(odds):
    """比分竞猜的「奖励倍数」-> 去水后的隐含概率。

    ⚠️ 平台给的「×2.36」这类数就是赔率（押 1 得 2.36），隐含概率 = 1/倍数，
    去水 = 按倒数之和归一。比分是 4 选 1，水位通常比两路胜负更高。
    返回 {"implied": {比分: 概率}, "vig": 水位, "n": 项数}；不足 2 项返回 None。
    """
    ok = {k: v for k, v in (odds or {}).items() if v and v > 1}
    if len(ok) < 2:
        return None
    inv = {k: 1.0 / v for k, v in ok.items()}
    s = sum(inv.values())
    return {"implied": {k: v / s for k, v in inv.items()},
            "vig": s - 1.0, "n": len(ok)}


def score_ev(dist, odds):
    """逐比分算期望收益：模型概率 × 倍数 − 1。返回 {比分: ev}。"""
    out = {}
    for k, v in (odds or {}).items():
        if not v or v <= 1:
            continue
        p = (dist or {}).get(k)
        if p is None:
            continue
        out[k] = p * v - 1.0
    return out


# ---------------------------------------------------------------- 期望比分
def _team_win_prob(p_map, need):
    """单局胜率 -> 整场胜率（先赢 need 局，各局独立同分布）"""
    return sum(math.comb(need - 1 + k, k) * p_map ** need * (1 - p_map) ** k
               for k in range(need))


def per_map_prob(p_win, best_of):
    """从「整场胜率」反解「单局胜率」（二分）。

    ⚠️ 为什么必须有这一步：`score_distribution` 要的是**单局胜率**。
    BO3 里 整场胜率 = p²(3−2p)，两者差很远 ——
    直接把整场胜率当单局用（2026-09-29 修掉的老 bug），会让比分分布隐含的整场胜率偏大：
    模型说 85%，分布隐含的其实是 **93.9%**，自相矛盾，而且横扫率被高估。
    """
    need = best_of // 2 + 1
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if _team_win_prob(mid, need) < p_win:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def score_distribution_from_win(p_win, best_of):
    """「整场胜率」-> 比分分布（自洽：各比分概率加总 == p_win）。

    比分预测应该用这个，不要直接拿整场胜率去喂 score_distribution。
    """
    return score_distribution(per_map_prob(p_win, best_of), best_of)


def score_distribution(p_map_a, best_of):
    """在「单局 A 胜率 = p_map_a、各局独立」的假设下，算出各种大比分的概率。

    best_of=3 就是 BO3（先赢 2 局）；best_of=5 是 BO5（先赢 3 局）。
    现实中各局不独立（地图 BAN/PICK、手感会传染），所以这只是个近似，
    但用来排「最可能的大比分」够用。
    """
    need = best_of // 2 + 1
    out = {}

    def rec(a, b, pr):
        if a == need or b == need:
            out[f"{a}:{b}"] = out.get(f"{a}:{b}", 0.0) + pr
            return
        rec(a + 1, b, pr * p_map_a)
        rec(a, b + 1, pr * (1 - p_map_a))

    rec(0, 0, 1.0)
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))
