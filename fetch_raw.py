# -*- coding: utf-8 -*-
"""
CFML 赛事数据采集（原始层）

把 PentaQ 开放接口的数据原样落盘到 raw/ 目录，不解析、不改写。
好处：抓一次就够，后面改解析逻辑不用重新抓。

用法：
    python fetch_raw.py            # 全量抓取（断点续传，已存在的文件会跳过）
    python fetch_raw.py --update   # 增量更新：刷新清单 + 只抓新比赛（日常用这个）
    python fetch_raw.py --meta     # 只抓赛季/轮次/对阵清单，不抓逐局明细
    python fetch_raw.py --detail   # 只补抓缺失的逐局明细

--update 会做三件事：
    1. 强制重新拉「赛季列表 / 轮次 / 对阵清单 / 当前赛季附表」——这些会随新比赛增长
    2. 抓取新出现的比赛明细
    3. 重抓「可能没采全」的比赛明细——判据是**数据完整性**而不是时间窗口（2026-09-30 改）：
       已采到的局数 < 该赛制打完所需的局数（BO3→2 / BO5→3 / BO7→4）就算半截数据，
       不管多老都继续追；再叠加最近 3 天的窗口兜底（明细是陆续放出的）。

数据源：https://cfm.admin.pentaq.com/api/
"""
import json
import os
import sys
import time
import gzip
import urllib.request
import urllib.error

from cfml_core import match_format, maps_needed   # 赛制识别只有这一份实现

BASE = "https://cfm.admin.pentaq.com/api/"
HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")
os.makedirs(RAW, exist_ok=True)

HDR = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Referer": "https://cfm.pentaq.com/",
    "Origin": "https://cfm.pentaq.com",
    "Accept": "application/json, text/plain, */*",
    "Accept-Encoding": "identity",
}

RETRY = 6
DELAY = 0.45         # 每次请求之间的间隔，别把人家服务器打挂（实测 0.25 会被 503 限流）
RECENT_DAYS = 3      # --update：最近 N 天内已开赛的比赛，明细兜底重抓（防陆续放出的明细）
MAX_TRIES = 3        # 同一场累计失败这么多次就不再重试（服务端永久缺数据的那些）

FORCE = False        # --update：让"会变化"的清单绕过本地缓存
_stats = {"req": 0, "hit": 0, "fail": 0, "new": 0, "dead": 0}
FAILED = {}          # {matchid: {tries, last, err}}，落盘在 raw/_failed.json


def api(path, retry=None):
    """带重试的 GET，返回解析后的 JSON。

    retry 可覆盖默认重试次数。
    503 特殊处理：绝大多数是"服务端就没这场的数据"，不是限流，所以只试 3 次就走，
    否则一次增量更新会在这几十场上卡半小时。
    """
    n_try = RETRY if retry is None else retry
    last = None
    for i in range(n_try):
        try:
            _stats["req"] += 1
            req = urllib.request.Request(BASE + path, headers=HDR)
            with urllib.request.urlopen(req, timeout=45) as r:
                raw = r.read()
            try:
                return json.loads(raw.decode("utf-8"))
            except Exception:
                return json.loads(gzip.decompress(raw).decode("utf-8"))
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (429, 500, 502, 503, 504):
                if e.code == 503 and i >= 2:
                    break          # 试 3 次还是 503，认定服务端没数据
                time.sleep(1.5 * (i + 1) if e.code == 503 else 3.0 * (i + 1))
                continue
            raise
        except Exception as e:
            last = e
            time.sleep(2.0 * (i + 1))
    _stats["fail"] += 1
    raise last


def save(name, obj):
    p = os.path.join(RAW, name)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    return p


def load(name):
    p = os.path.join(RAW, name)
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- 第一层：赛季
def fetch_contests():
    d = None if FORCE else load("_contests.json")
    if d:
        _stats["hit"] += 1
        return d
    r = api("matrixData/getmatchList")
    contests = r["data"]["contest"]
    save("_contests.json", contests)
    time.sleep(DELAY)
    return contests


# ---------------------------------------------------------------- 第二层：轮次
def fetch_subcontests(contestid):
    name = f"sub_{contestid}.json"
    d = None if FORCE else load(name)
    if d is not None:
        _stats["hit"] += 1
        return d
    r = api(f"matrixData/getmatchList?contestid={contestid}")
    subs = (r.get("data") or {}).get("subcontest") or []
    save(name, subs)
    time.sleep(DELAY)
    return subs


# ---------------------------------------------------------------- 第三层：对阵
def fetch_matches(contestid, subcontestid):
    name = f"matches_{contestid}_{subcontestid}.json"
    d = None if FORCE else load(name)
    if d is not None:
        _stats["hit"] += 1
        return d
    r = api(f"matrixData/getmatchList?contestid={contestid}&subcontestid={subcontestid}")
    ms = (r.get("data") or {}).get("matchList") or []
    save(name, ms)
    time.sleep(DELAY)
    return ms


# ---------------------------------------------------------------- 第四层：逐局明细
def fetch_detail(matchid, refresh=False):
    """抓一场的逐局明细。

    返回值：'write' 新写入 / 'skip' 本地已有 / 'empty' 服务端暂无数据（比赛没打完）
            / 'dead' 屡试屡败，已放弃
    refresh=True 时即使本地已有也重抓（用于最近刚开赛的比赛，防止半截数据）。
    """
    name = f"game_{matchid}.json"
    p = os.path.join(RAW, name)
    if os.path.exists(p) and not refresh:
        _stats["hit"] += 1
        return "skip"

    key = str(matchid)
    rec = FAILED.get(key)
    # 已经被证实拿不到的（服务端缺数据），别再浪费时间；除非这次是主动重抓
    if rec and rec.get("tries", 0) >= MAX_TRIES and not refresh:
        _stats["dead"] += 1
        return "dead"

    try:
        r = api(f"authApi/getGameOverData?game_id={matchid}",
                retry=2 if rec else None)
    except Exception as e:
        rec = FAILED.setdefault(key, {"tries": 0})
        rec["tries"] = rec.get("tries", 0) + 1
        rec["last"] = time.strftime("%Y-%m-%d %H:%M:%S")
        rec["err"] = str(e)[:100]
        raise

    body = r.get("data") if isinstance(r, dict) else None
    # 比赛还没打完 / 服务端没有这场的数据 → 不要落盘，否则半截数据会被永久固化
    if not isinstance(body, dict) or not [k for k in body if str(k).strip()]:
        FAILED.pop(key, None)      # 不算失败，是还没打
        time.sleep(DELAY)
        return "empty"
    save(name, r)
    FAILED.pop(key, None)          # 拿到了就从失败名单里移除
    time.sleep(DELAY)
    return "write"


# ---------------------------------------------------------------- 赛季级附表
def fetch_season_extras(contestid):
    """每个赛季抓两份附表：
    _players_<cid>.json  —— 选手赛季数据（含出场回合数，用来判断轮换）
    _teams_<cid>.json    —— 战队赛季数据（含胜率等）
    """
    for tag, path in (("players", f"matrixData/getplayerData?contestid={contestid}"),
                      ("teams", f"matrixData/getteamData?contestid={contestid}")):
        name = f"_{tag}_{contestid}.json"
        if os.path.exists(os.path.join(RAW, name)) and not FORCE:
            _stats["hit"] += 1
            continue
        r = api(path)
        save(name, r.get("data") if isinstance(r, dict) else r)
        time.sleep(DELAY)


def fetch_overview():
    """当前赛季总览（就是虎牙面板上那一屏）"""
    name = "_overview.json"
    if os.path.exists(os.path.join(RAW, name)) and not FORCE:
        _stats["hit"] += 1
        return
    r = api("authApi/searchGameData")
    save(name, r.get("data") if isinstance(r, dict) else r)
    time.sleep(DELAY)


def main():
    global FORCE
    argv = set(sys.argv[1:])
    want_detail = "--meta" not in argv
    meta_only = "--detail" in argv
    update_mode = "--update" in argv
    if update_mode:
        FORCE = True

    log("=" * 64)
    log("CFML 数据采集（原始层）" + ("   [增量更新模式]" if update_mode else ""))
    log("=" * 64)

    FAILED.update(load("_failed.json") or {})
    if FAILED:
        log(f"已知拿不到的场次 {len(FAILED)} 场（累计失败 {MAX_TRIES} 次以上的直接跳过）")

    contests = fetch_contests()
    log(f"赛季 {len(contests)} 个")
    fetch_overview()
    for c in contests:
        fetch_season_extras(c["contestid"])
    log("赛季附表已就绪\n")

    # ---- 第一遍：把赛季 / 轮次 / 对阵清单全部拉下来，顺便收集 matchid
    index = []
    for c in contests:
        cid, cname = c["contestid"], c["contestname"]
        subs = fetch_subcontests(cid)
        n_match = 0
        for s in subs:
            sid, sname = s["subcontestid"], s["subcontestname"]
            ms = fetch_matches(cid, sid)
            for m in ms:
                index.append({
                    "contestid": cid, "contestname": cname,
                    "subcontestid": sid, "subcontestname": sname,
                    "matchid": m.get("matchid"),
                    "teamA": m.get("teamA"), "teamB": m.get("teamB"),
                    "time": m.get("time"),
                })
            n_match += len(ms)
        log(f"  {cname:<20} 轮次 {len(subs):>2}  对阵 {n_match:>4}")

    save("_index.json", index)
    log(f"\n对阵总数：{len(index)}")

    # 最近 N 天内「已开赛」的，明细允许重抓（防半截数据）。
    # 注意要同时卡上界：赛程里未来的比赛时间也满足 >= cutoff，那些返回空数据属于白跑。
    now_str = time.strftime("%Y-%m-%d %H:%M:%S")
    recent_cutoff = time.strftime("%Y-%m-%d %H:%M:%S",
                                  time.localtime(time.time() - RECENT_DAYS * 86400))

    def is_recent(t):
        return recent_cutoff <= (t or "") <= now_str

    def maps_in_local_file(mid):
        """本地已落盘的明细里有几局（0 = 没有文件或解析不了）。"""
        p = os.path.join(RAW, f"game_{mid}.json")
        if not os.path.exists(p):
            return 0
        try:
            with open(p, encoding="utf-8") as f:
                body = json.load(f).get("data") or {}
            return sum(1 for k in body if str(k).strip())
        except Exception:
            return 0

    def needs_refresh(x):
        """--update 时这场要不要强制重抓明细。

        判据（2026-09-30 从「时间窗口」改成「数据完整性」）：
          a. 最近 N 天已开赛 → 重抓（兜底：明细陆续放出，采全了的重复抓一次也不亏）
          b. 更早的比赛：本地文件里的局数 < 该赛制打完所需局数 → 重抓
             （半截数据不管多老都追。之前按纯时间窗口，XROCK vs EP 卡在 1 局 4 天没人管）
        """
        t = x.get("time") or ""
        if is_recent(t):
            return True
        if not t or t > now_str:          # 还没开赛的，走正常抓取路径
            return False
        mid = x.get("matchid")
        have = maps_in_local_file(mid)
        if have == 0:                     # 本地没有文件，谈不上"半截"，正常抓
            return False
        need = maps_needed(match_format(x.get("subcontestname"), x.get("contestname")))
        return have < need

    if update_mode:
        n_recent = sum(1 for x in index if is_recent(x.get("time")))
        n_half = sum(1 for x in index
                     if x.get("matchid") and not is_recent(x.get("time"))
                     and (x.get("time") or "") <= now_str
                     and needs_refresh(x))
        log(f"最近 {RECENT_DAYS} 天内已开赛 {n_recent} 场、更早的半截数据 {n_half} 场，"
            f"明细将强制重抓")

    if not want_detail:
        log("（--meta：已跳过分级明细抓取）")
        return

    # ---- 第二遍：逐场抓逐局明细
    todo = [x for x in index if x.get("matchid")]
    # 去重（同一 matchid 可能被多个轮次引用）
    seen, uniq = set(), []
    for x in todo:
        if x["matchid"] in seen:
            continue
        seen.add(x["matchid"])
        uniq.append(x)
    log(f"待抓明细：{len(uniq)} 场（去重后）\n")

    if meta_only:
        done = sum(1 for x in uniq if os.path.exists(os.path.join(RAW, f'game_{x["matchid"]}.json')))
        log(f"（--detail：已有 {done}/{len(uniq)} 份，现在补抓缺失的）")

    t0 = time.time()
    n_empty = n_fail = 0
    for i, x in enumerate(uniq, 1):
        mid = x["matchid"]
        refresh = update_mode and is_recent(x.get("time"))
        try:
            st = fetch_detail(mid, refresh=refresh)
        except Exception as e:
            n_fail += 1
            log(f"  ! 失败 matchid={mid} {x['teamA']} vs {x['teamB']}: {e}")
            continue
        if st == "write":
            _stats["new"] += 1
        elif st == "empty":
            n_empty += 1

        if i % 50 == 0 or i == len(uniq):
            el = time.time() - t0
            eta = el / i * (len(uniq) - i) if i else 0
            log(f"  进度 {i}/{len(uniq)}  已用 {el/60:.1f} 分  预计还需 {eta/60:.1f} 分")

    raw_files = [f for f in os.listdir(RAW) if f.startswith("game_")]
    log(f"\n完成。raw/ 下明细文件 {len(raw_files)} 份")
    if update_mode:
        log(f"本次新增/刷新明细 {_stats['new']} 份")
    log(f"请求 {_stats['req']} 次（命中缓存 {_stats['hit']}，失败 {_stats['fail']}）")
    if _stats["dead"]:
        log(f"跳过已知拿不到的场次 {_stats['dead']} 场（服务端缺数据，不再重试）")
    if n_empty:
        log(f"服务端暂无数据 {n_empty} 场（通常是还没打完）——已跳过，未落盘")
    if n_fail:
        log(f"本次新失败 {n_fail} 场（累计 {MAX_TRIES} 次后会自动跳过）")

    save("_failed.json", FAILED)
    log(f"失败名单已更新：{len(FAILED)} 场（存在 raw/_failed.json）")


if __name__ == "__main__":
    main()
