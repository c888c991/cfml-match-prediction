# -*- coding: utf-8 -*-
"""
一键重建数据链路：raw/ 下的原始数据 → cfml.db 全部就绪。

    python rebuild_all.py

等价于按顺序跑：
    1. build_db.py                原始 JSON → SQLite（⚠️ 会冲掉 round_snap）
    2. build_round_snapshots.py   逐回合流水 → round_snap（实时预测地基）
    3. build_features.py          预测点 + 特征表

之所以要这个脚本：build_db.py 是整库重建，每次都会把 round_snap 表冲掉，
历史上踩过 3 次「重建完忘记补快照表」的坑（最近一次 2026-09-30）。
以后重建一律用它，别再单独跑 build_db.py。
"""
import subprocess
import sys
import time
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

STEPS = [
    ("build_db.py", "原始 JSON → SQLite（会冲掉 round_snap）"),
    ("build_round_snapshots.py", "逐回合流水 → round_snap 快照表"),
    ("build_features.py", "预测点 + 特征表（含卡尔曼两列）"),
]


def main():
    print("=" * 72)
    print("一键重建：build_db → round_snapshots → features")
    print("=" * 72)
    t0 = time.time()
    for i, (script, desc) in enumerate(STEPS, 1):
        print(f"\n[{i}/{len(STEPS)}] {script} —— {desc}")
        print("-" * 72)
        t = time.time()
        r = subprocess.run([PY, "-u", os.path.join(HERE, script)], cwd=HERE)
        if r.returncode != 0:
            print(f"\n✗ {script} 失败（exit {r.returncode}），停止后续步骤。")
            print("  注意：此时 round_snap 很可能已被冲掉，修好后重跑本脚本即可。")
            sys.exit(r.returncode)
        print(f"（{script} 用时 {time.time()-t:.0f} 秒）")

    # ---- 收尾自检：三张表都在、都有数 ----
    import sqlite3
    c = sqlite3.connect(os.path.join(HERE, "cfml.db"))
    ok = True
    for tbl in ("match", "map", "round_snap", "feature_point", "player_map"):
        try:
            n = c.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
            flag = "✓" if n else "✗ 空！"
            if not n:
                ok = False
            print(f"  {tbl:<16}{n:>9}  {flag}")
        except sqlite3.OperationalError:
            print(f"  {tbl:<16}    缺表 ✗")
            ok = False
    c.close()
    print("-" * 72)
    print(f"完成，总用时 {(time.time()-t0)/60:.1f} 分钟。"
          + ("全部就绪 ✓" if ok else "有表是空的，检查上面的 ✗"))
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
