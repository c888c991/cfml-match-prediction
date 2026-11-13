# -*- coding: utf-8 -*-
"""predict.py —— CFML 预测命令行（两个模式）

    ┌─────────────────────────────────────────────────────────────┐
    │  模式一：局前预测（prematch）                                │
    │    赛前独立时点，看不到任何本场信息                          │
    │    python predict.py prematch --team-a XROCK.快手 --team-b EP.快手
    │                                                             │
    │  模式二：实时预测（live）                                    │
    │    拿着当前比分滚动预测（比分从截屏/人工录入）               │
    │    python predict.py live --team-a XROCK.快手 --team-b EP.快手 \
    │         --score 0:1 --map-round 10 --map-score 4:6          │
    └─────────────────────────────────────────────────────────────┘

两个模式共用 cfml_core 的内核，所以口径一致。
加上 --json 就输出纯 JSON（给别的程序调）；不加就打印人看的版本。
加上 --odds-a / --odds-b 会顺便和「市场共识的隐含概率」对照，看模型判断与市场分歧多大。
加上 --score-odds 可以把模型给出的比分分布，与市场对比分结果的定价逐项对照。

详细输入输出契约见同目录的 预测接口契约.md。
"""
import argparse
import json
import os
import sys
import sqlite3
from datetime import datetime

from cfml_core import (connect, load_all, slice_history, prematch_feats, make_row,
                       featurize, train_on_feature_points, sigmoid_predict,
                       fit_round_market, pred_round_market,
                       fit_lead_model, pred_map_from_lead,
                       odds_to_prob, ev_of_bet, score_distribution,
                       score_distribution_from_win, per_map_prob,
                       parse_score_odds, score_odds_to_prob, score_ev,
                       kalman_strength, kalman_feats, kalman_state_feats,
                       is_playoff_of, match_format, maps_needed,
                       FEATS, FEAT_DOC, norm_team)

HERE = os.path.dirname(os.path.abspath(__file__))
VERSION = "1.0"


def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------- 找比赛
def find_match(c, a, b, near_time=None, prefer_future=False):
    """在库里找 a vs b 那场比赛（两边顺序无所谓），返回 dict 或 None

    near_time 给了就取时间上最接近的那场；
    prefer_future=True 时优先取「还没开打、离现在最近」的那场（预测下一场就该这样）。
    """
    rows = c.execute("""
        SELECT matchid, contestid, contestname, subcontestname, match_time,
               teamA, teamB, n_maps, winner, scoreA, scoreB, has_detail
        FROM match
        WHERE ((teamA=? AND teamB=?) OR (teamA=? AND teamB=?))
          AND is_official=1 AND is_placeholder=0
        ORDER BY match_time DESC
    """, (a, b, b, a)).fetchall()
    if not rows:
        return None
    rows = [dict(r) for r in rows]
    if near_time:
        best, bd = None, None
        for r in rows:
            try:
                d = abs((datetime.strptime(r["match_time"][:19], "%Y-%m-%d %H:%M:%S")
                         - datetime.strptime(near_time[:19], "%Y-%m-%d %H:%M:%S")).total_seconds())
            except Exception:
                continue
            if bd is None or d < bd:
                best, bd = r, d
        if best is not None:
            return best
    if prefer_future:
        now = datetime.now()
        recent = now.timestamp() - 7 * 86400
        cand = []
        for r in rows:
            try:
                t = datetime.strptime(r["match_time"][:19], "%Y-%m-%d %H:%M:%S")
            except Exception:
                continue
            # 「还没定结果」的算下一场：时间还没到；或者刚刚打完、明细还没采到（近 7 天内）。
            # 注意别把 2021 那种「本来就没明细」的老场次算进来。
            if t >= now or (not r.get("has_detail") and t.timestamp() >= recent):
                cand.append(r)
        if cand:
            return min(cand, key=lambda r: r["match_time"])
    return rows[0]


def current_season(c):
    r = c.execute("""
        SELECT contestid, contestname FROM contest
        ORDER BY contestid DESC LIMIT 1
    """).fetchone()
    return (r["contestid"], r["contestname"]) if r else (None, None)


def live_from_db(c, mid, maps_played):
    """如果这场在库里已有明细（回放场景），把已完成局的真实数据取出来"""
    ms = c.execute("SELECT teamA FROM match WHERE matchid=?", (mid,)).fetchone()
    if not ms:
        return {}
    a = ms["teamA"]
    maps = [dict(r) for r in c.execute(
        "SELECT map_seq, scoreA, scoreB, winner FROM map WHERE matchid=? ORDER BY map_seq", (mid,))]
    done = maps[:maps_played]
    if not done:
        return {}
    rnd = sum((g["scoreA"] or 0) - (g["scoreB"] or 0) for g in done)
    last = done[-1]["winner"]
    return {"rnd_diff": rnd, "last_map_won": 1 if last == a else 0}


# ---------------------------------------------------------------- 组装输出
def get_score_odds(args):
    """取 args.score_odds，取不到就当没给。

    ⚠️ selfcheck.py / prematch_board.py 是**手工拼 Namespace** 调 mode_prematch 的，
    新加的参数它们不会有 → 必须用 getattr，否则一加参数就把它们全打挂。
    """
    return parse_score_odds(getattr(args, "score_odds", None))


def build_output(mode, m, a, b, asof, pre, live, p_match, p_map, rounds_over,
                 match_row, extra_state, odds, score_odds=None):
    # 比分分布必须按**赛制**（BO 几）展开，不能用 n_maps：
    # BO5 横扫 3:0 时 n_maps=3 会被误判成 BO3；未开赛时 n_maps=0。
    # 2026-09-30 起改用 match_format（实测比分对数损失改善 10.1%，见 exp_score.py）。
    best_of = match_format((match_row or {}).get("subcontestname"),
                           (match_row or {}).get("contestname"))
    # ⚠️ 比分分布要的是**整场胜率**（p_match），走 score_distribution_from_win 内部会
    # 反解单局胜率（见 cfml_core.per_map_prob 的说明）。
    # 而 p_map 的语义是**单局/下一局**谁赢，两者不是一回事：
    #   BO3 里 p_match = p_map²(3−2p_map)，整场胜率天然比单局更极端。
    # 2026-09-30 修的 bug：以前不分场合都把 p_match 当 p_map 传，
    #   结果 ①比分分布多反解了一次（分布隐含的整场胜率对不上）
    #        ②map_winner（第 1 局谁赢）被报成了整场胜率，低估 5 个百分点左右。
    dist = score_distribution_from_win(p_match, best_of)
    top = [{"score": k, "p": round(v, 4)} for k, v in list(dist.items())[:5]]

    out = {
        "mode": mode,
        "version": VERSION,
        "generated_at": now_str(),
        "match": {
            "teamA": a, "teamB": b,
            "match_time": (match_row or {}).get("match_time"),
            "contestname": (match_row or {}).get("contestname"),
            "matchid": (match_row or {}).get("matchid"),
            "n_maps": (match_row or {}).get("n_maps"),   # 实际打了几局（未开赛是 0）
            "best_of": best_of,                          # 赛制：比分分布按它展开
        },
        "asof": asof,
        "state": extra_state,
        "predict": {
            "match_winner": {"A": round(p_match, 4), "B": round(1 - p_match, 4)},
            "map_winner": {"A": round(p_map, 4), "B": round(1 - p_map, 4)},
            "final_score_top5": top,
            "rounds_over_16_5": round(rounds_over, 4),
            "rounds_under_16_5": round(1 - rounds_over, 4),
        },
        "evidence": {
            "A": {"衰减加权胜率": pre.get("a_wr"), "本赛季胜率": pre.get("a_sznwr"),
                  "衰减加权单局胜率": pre.get("a_mapwr"), "有效历史场次": pre.get("a_n")},
            "B": {"衰减加权胜率": pre.get("b_wr"), "本赛季胜率": pre.get("b_sznwr"),
                  "衰减加权单局胜率": pre.get("b_mapwr"), "有效历史场次": pre.get("b_n")},
            "交手": {"A 胜率": pre.get("h2h_wr"), "场次": pre.get("h2h_n"),
                     "上次交手 A 是否赢": pre.get("h2h_last")},
            "胜率差绝对值": (abs((pre.get("a_wr") or 0.5) - (pre.get("b_wr") or 0.5))
                          if pre.get("a_wr") is not None and pre.get("b_wr") is not None else None),
        },
        "model": {"name": "逻辑回归", "features": len(FEATS), "n_train_points": m.get("n", 0)},
    }
    if odds:
        imp = odds_to_prob(odds.get("a"), odds.get("b"))
        if imp:
            out["market"] = {
                "odds_A": odds.get("a"), "odds_B": odds.get("b"),
                "implied_A": round(imp["implied_A"], 4),
                "implied_B": round(imp["implied_B"], 4),
                "water": round(imp["vig"], 4),
                "model_minus_market_A": round(p_match - imp["implied_A"], 4),
                "ev_bet_A": (round(ev_of_bet(p_match, odds.get("a")), 4)
                             if odds.get("a") else None),
                "ev_bet_B": (round(ev_of_bet(1 - p_match, odds.get("b")), 4)
                             if odds.get("b") else None),
            }
    if score_odds:
        sp = score_odds_to_prob(score_odds)
        if sp:
            ev = score_ev(dist, score_odds)
            out["market_score"] = {
                "odds": dict(score_odds),
                "implied": {k: round(v, 4) for k, v in sp["implied"].items()},
                "water": round(sp["vig"], 4),
                "model": {k: round(dist.get(k, 0.0), 4) for k in score_odds},
                "model_minus_market": {k: round(dist.get(k, 0.0)
                                                - sp["implied"].get(k, 0.0), 4)
                                       for k in score_odds},
                "ev": {k: round(v, 4) for k, v in ev.items()},
                "best": (max(ev, key=ev.get) if ev else None),
            }
    return out


def print_human(out):
    m = out["match"]
    p = out["predict"]
    print("=" * 66)
    print(f"{m['teamA']}  vs  {m['teamB']}")
    meta = []
    if m.get("contestname"):
        meta.append(m["contestname"])
    if m.get("match_time"):
        meta.append(m["match_time"])
    if m.get("matchid"):
        meta.append(f"matchid {m['matchid']}")
    print("  " + " · ".join(meta) if meta else "")
    print(f"  模式 = {out['mode']}   截断时点 = {out['asof']}")
    if out.get("state"):
        s = out["state"]
        bits = []
        if s.get("maps_played") is not None:
            bits.append(f"大比分 {s.get('scoreA')}:{s.get('scoreB')}")
        if s.get("cur_map_round") is not None:
            bits.append(f"当前局第 {s['cur_map_round']} 回合 {s.get('cur_map_a')}:{s.get('cur_map_b')}")
        if bits:
            print("  当前状态 = " + "，".join(bits))
    print("-" * 66)
    print(f"  本场胜率      {m['teamA']} {p['match_winner']['A']*100:5.1f}%"
          f"   |   {m['teamB']} {p['match_winner']['B']*100:5.1f}%")
    tag = "当前局" if out["mode"] == "live" else "第 1 局"
    print(f"  {tag}胜率    {m['teamA']} {p['map_winner']['A']*100:5.1f}%"
          f"   |   {m['teamB']} {p['map_winner']['B']*100:5.1f}%")
    print(f"  单局回合数    大于 16.5  {p['rounds_over_16_5']*100:5.1f}%"
          f"   |   小于 16.5  {p['rounds_under_16_5']*100:5.1f}%")
    print(f"  最可能的大比分：")
    for it in p["final_score_top5"][:3]:
        print(f"      {it['score']}   {it['p']*100:5.1f}%")
    e = out["evidence"]

    def _f(v, pct=False):
        """None 显示成破折号；概率统一到 3 位小数或百分比，别甩一长串浮点"""
        if v is None:
            return "—"
        return f"{v*100:.1f}%" if pct else f"{v:.3f}"

    print("-" * 66)
    print(f"  依据  衰减加权胜率 {m['teamA']} {_f(e['A']['衰减加权胜率'])}"
          f"   {m['teamB']} {_f(e['B']['衰减加权胜率'])}")
    print(f"        （含全部历史、越近权重越高，半衰期 365 天）")
    print(f"        本赛季胜率   {m['teamA']} {_f(e['A']['本赛季胜率'], True)}"
          f"   {m['teamB']} {_f(e['B']['本赛季胜率'], True)}")
    print(f"        历史交手     {e['交手']['场次']} 场，"
          f"{m['teamA']} 胜率 {_f(e['交手']['A 胜率'], True)}")
    if out.get("market"):
        k = out["market"]
        print("-" * 66)
        print(f"  盘口对比  庄家赔率 {k['odds_A']} / {k['odds_B']}")
        print(f"            庄家隐含 {k['implied_A']*100:.1f}% / {k['implied_B']*100:.1f}%"
              f"   （水位 {k['water']*100:.1f}%）")
        print(f"            模型比庄家  {k['model_minus_market_A']*100:+.1f} 个百分点"
              f"   （{m['teamA']} 方向）")
        if k.get("ev_bet_A") is not None:
            print(f"            押 1 元的期望收益：{m['teamA']} {k['ev_bet_A']:+.3f} 元"
                  f"   {m['teamB']} {k['ev_bet_B']:+.3f} 元")
            print("            （正数才值得押；这个数没算模型误差，别当铁律）")
    if out.get("market_score"):
        s = out["market_score"]
        print("-" * 66)
        print(f"  比分竞猜对比　水位 {s['water']*100:.1f}%"
              f"（平台 4 选 1，抽水通常比胜负高）")
        print(f"      {'比分':<6}{'平台倍数':>9}{'隐含概率':>10}{'模型概率':>10}{'模型−平台':>11}   期望收益")
        for k, od in s["odds"].items():
            mk = s["model"].get(k) or 0.0
            im = s["implied"].get(k) or 0.0
            ev = s["ev"].get(k)
            evs = f"{ev:+.3f} 元" if ev is not None else "—"
            print(f"      {k:<6}{od:>9.2f}{im*100:>9.1f}%{mk*100:>9.1f}%"
                  f"{(mk-im)*100:>+10.1f}pp   {evs}")
        if s.get("best"):
            print(f"      期望收益最高的一项：{s['best']}"
                  f"（{s['ev'][s['best']]:+.3f} 元）")
        print("      （这一局比分是 4 选 1，最可能比分本身命中率上限只有约 49%）")
    print(f"  模型用 {out['model']['n_train_points']} 个预测点训练")
    print("=" * 66)


# ---------------------------------------------------------------- 卡尔曼特征
def has_kf_columns(c):
    """feature_point 表里有没有卡尔曼那两列（没重建过特征表就没有）"""
    return any(r[1] == "kf_diff" for r in c.execute("PRAGMA table_info(feature_point)"))


def kalman_into_row(r, ms, asof, row):
    """把卡尔曼特征填进 r['kf_diff'] / r['kf_conf']。

    库里查得到这场 → 用它「开赛前」的状态；
    查不到（还没开赛）→ 用截至 asof 的最终状态。

    ⚠️ 不要按 swapped 取负！r 里的特征全都是**库里 teamA/teamB 方向**的
    （prematch_feats 拿到的就是 fa/fb），用户方向只在最后输出 evidence 时才翻。
    这里跟着翻会让「两队调换」算出不同的概率，A/B 就不再互补了。
    """
    kf, state = kalman_strength(ms, asof=asof)
    mid = (row or {}).get("matchid")
    kfd, kfc = kalman_feats(kf, mid) if mid else (None, None)
    if kfd is None:
        kfd, kfc = kalman_state_feats(state, (row or {}).get("teamA"),
                                      (row or {}).get("teamB"))
    if kfd is None:
        return False
    r["kf_diff"] = kfd
    r["kf_conf"] = kfc
    return True


# ---------------------------------------------------------------- 方向对齐
def orient(row, a, b):
    """模型是按「赛程里排在左边那支（库里 teamA）」训练的。

    如果用户传进来的 A 刚好是库里的 teamB，特征就必须按库里方向算，
    算完再把概率翻回来。返回 (特征用A方, 特征用B方, 是否颠倒)。
    """
    dba, dbb = (row or {}).get("teamA"), (row or {}).get("teamB")
    if dbb and dba and dba == b and dbb == a:
        return b, a, True
    return a, b, False


def flip_pre(pre):
    """把赛前特征表从「库里方向」翻成「用户方向」"""
    out = dict(pre)
    for k in ("wr", "mapwr", "netmaps", "rd", "n", "sznwr"):
        out["a_" + k], out["b_" + k] = pre.get("b_" + k), pre.get("a_" + k)
    for k in ("h2h_wr", "h2h_last"):
        v = pre.get(k)
        out[k] = None if v is None else 1 - v
    return out


# ---------------------------------------------------------------- 两个模式
def mode_prematch(args):
    c = connect()
    a, b = norm_team(args.team_a), norm_team(args.team_b)
    row = find_match(c, a, b, near_time=args.time, prefer_future=True) if not args.time \
        else find_match(c, a, b, near_time=args.time)
    if row is None:
        cid, cname = current_season(c)
        row = {"matchid": None, "contestid": cid, "contestname": cname,
               "match_time": args.time, "n_maps": 3}
    asof = args.time or row.get("match_time") or now_str()

    fa, fb, swapped = orient(row, a, b)
    ms, maps_of, deaths = load_all(c)
    hist, h2h, szn = slice_history(ms, maps_of, asof=asof)
    pre_db = prematch_feats(hist, h2h, szn, fa, fb, asof, row.get("contestid"))
    r = make_row(pre_db, live=None, mapseq=0,
                 is_playoff=is_playoff_of(row.get("subcontestname")))
    if has_kf_columns(c):
        kalman_into_row(r, ms, asof, row)
    model, n = train_on_feature_points(c, asof=asof)
    p_db = sigmoid_predict(model, featurize(r))
    p_match = (1 - p_db) if swapped else p_db
    pre = flip_pre(pre_db) if swapped else pre_db

    rounds = pred_round_market(fit_round_market(c),
                               abs((pre.get("a_wr") or 0.5) - (pre.get("b_wr") or 0.5)))
    # 第 1 局胜率要由整场胜率按**赛制**反解（BO3 里两者差 5 个百分点左右），
    # 不能直接拿整场胜率当单局，也不能拿 n_maps 当赛制。
    out = build_output("prematch", model, a, b, asof, pre, None, p_match,
                       per_map_prob(p_match, match_format(row.get("subcontestname"),
                                                          row.get("contestname"))),
                       rounds, row, None,
                       {"a": args.odds_a, "b": args.odds_b} if args.odds_a or args.odds_b else None,
                       get_score_odds(args))
    c.close()
    return out


def mode_live(args):
    c = connect()
    a, b = norm_team(args.team_a), norm_team(args.team_b)
    row = find_match(c, a, b, near_time=args.start_time) if args.start_time \
        else find_match(c, a, b, prefer_future=True)
    if row is None:
        cid, cname = current_season(c)
        row = {"matchid": None, "contestid": cid, "contestname": cname,
               "match_time": args.start_time, "n_maps": 3}

    asof = args.start_time or row.get("match_time") or now_str()
    sa, sb = parse_score(args.score) if args.score else (0, 0)
    maps_played = sa + sb

    fa, fb, swapped = orient(row, a, b)
    ms, maps_of, deaths = load_all(c)
    hist, h2h, szn = slice_history(ms, maps_of, asof=asof)
    pre_db = prematch_feats(hist, h2h, szn, fa, fb, asof, row.get("contestid"))

    sign = -1 if swapped else 1
    live = {"score_diff": (sa - sb) * sign, "mapseq": maps_played}
    if args.rnd_diff is not None:
        live["rnd_diff"] = args.rnd_diff * sign
    # 只有在明确要求「回放」时才去库里读已打完那几局的真实数据。
    # 默认不读 —— 否则会跟你手动录入的比分打架。
    if args.use_db_state and row.get("matchid"):
        live.update(live_from_db(c, row["matchid"], maps_played) or {})

    r = make_row(pre_db, live=live, mapseq=maps_played,
                 is_playoff=is_playoff_of(row.get("subcontestname")))
    if has_kf_columns(c):
        kalman_into_row(r, ms, asof, row)
    model, n = train_on_feature_points(c, asof=asof)
    p_db = sigmoid_predict(model, featurize(r))
    p_match = (1 - p_db) if swapped else p_db
    pre = flip_pre(pre_db) if swapped else pre_db

    # 当前局胜率：能读到「打到第几回合、比分多少」就用局中领先模型，
    # 否则退回「由整场胜率反解的单局胜率」（不能直接拿整场胜率当单局用）。
    lead = None
    if args.map_round is not None and args.map_score:
        ma, mb = parse_score(args.map_score)
        lead = ma - mb
        p_map_db = pred_map_from_lead(fit_lead_model(c), args.map_round, lead * sign)
        p_map = (1 - p_map_db) if swapped else p_map_db
    else:
        p_map = per_map_prob(p_match, match_format(row.get("subcontestname"),
                                                   row.get("contestname")))

    rounds = pred_round_market(fit_round_market(c),
                               abs((pre.get("a_wr") or 0.5) - (pre.get("b_wr") or 0.5)))
    ma = mb = None
    if args.map_score:
        ma, mb = parse_score(args.map_score)
    state = {"maps_played": maps_played, "scoreA": sa, "scoreB": sb,
             "cur_map_round": args.map_round, "cur_map_a": ma, "cur_map_b": mb,
             "cur_map_name": args.map_name, "cur_map_lead": lead,
             "source": args.source or "manual"}
    out = build_output("live", model, a, b, asof, pre, live, p_match, p_map, rounds,
                       row, state,
                       {"a": args.odds_a, "b": args.odds_b} if args.odds_a or args.odds_b else None,
                       get_score_odds(args))
    c.close()
    return out


def parse_score(s):
    if not s:
        return (0, 0)
    for sep in (":", "：", "-"):
        if sep in s:
            x, y = s.split(sep, 1)
            return (int(x.strip() or 0), int(y.strip() or 0))
    raise SystemExit(f"比分格式不对：{s}（应该是 1:0 这样）")


def build_parser():
    ap = argparse.ArgumentParser(description="CFML 预测：局前 / 实时两个模式")
    sub = ap.add_subparsers(dest="mode", required=True)

    def common(p):
        p.add_argument("--team-a", required=True, help="A 方队名（赛事页左边那支）")
        p.add_argument("--team-b", required=True, help="B 方队名")
        p.add_argument("--odds-a", type=float, help="盘口上 A 的赔率（可选）")
        p.add_argument("--odds-b", type=float, help="盘口上 B 的赔率（可选）")
        p.add_argument("--score-odds", dest="score_odds", default=None,
                       help='比分竞猜的倍数，如 "2:0=6.20,2:1=3.81,1:2=2.34,0:2=2.07"（可选）')
        p.add_argument("--json", action="store_true", help="只输出 JSON")
        p.add_argument("--save", help="把 JSON 存到文件")

    p1 = sub.add_parser("prematch", help="局前预测：赛前独立时点")
    common(p1)
    p1.add_argument("--time", help="开赛时间 YYYY-MM-DD HH:MM:SS（不填取赛程或当前时间）")

    p2 = sub.add_parser("live", help="实时预测：拿当前比分滚动预测")
    common(p2)
    p2.add_argument("--start-time", help="开赛时间（用来截断赛前特征，强烈建议填）")
    p2.add_argument("--score", default="0:0", help="当前大比分，如 0:1")
    p2.add_argument("--map-round", type=int, help="当前局打到第几回合（1~16）")
    p2.add_argument("--map-score", help="当前局回合比分，如 4:6")
    p2.add_argument("--map-name", help="当前地图名（可选，只做记录）")
    p2.add_argument("--rnd-diff", type=int, help="已打完那几局的回合总分差（可选，填了更准）")
    p2.add_argument("--use-db-state", action="store_true",
                    help="回放用：从库里读已打完几局的真实数据（默认不读，避免跟手输比分打架）")
    p2.add_argument("--source", help="数据来源标注，如 screenshot（可选）")
    return ap


def main():
    args = build_parser().parse_args()
    out = mode_prematch(args) if args.mode == "prematch" else mode_live(args)
    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print_human(out)
        if args.save:
            print(f"已保存 JSON → {args.save}")


if __name__ == "__main__":
    main()
