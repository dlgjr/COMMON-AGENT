#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq


def write_jsonl(rows, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[DATA] {len(rows)} rows -> {output}")


def awm_rows(root):
    root = Path(root)
    path = root / "gen_tasks.jsonl"
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)
            scenario = x.get("scenario", "")
            for task_idx, task in enumerate(x.get("tasks", [])):
                rows.append({
                    "messages": [{"role": "user", "content": "<environment task>"}],
                    "env_config": {
                        "name": "common_agent",
                        "kind": "awm",
                        "data_root": str(root),
                        "scenario": scenario,
                        "task_idx": task_idx,
                        "task": task,
                    },
                })
    return rows


def alfworld_rows(root):
    root = Path(root)
    rows = []
    for game_file in sorted(root.glob("**/game.tw-pddl")):
        rows.append({
            "messages": [{"role": "user", "content": "<environment task>"}],
            "env_config": {
                "name": "common_agent",
                "kind": "alfworld",
                "game_file": str(game_file),
                "episode_steps": 64,
            },
        })
    return rows


def find_webshop_items(root):
    root = Path(root)
    for pattern in (
        "**/items_shuffle_1000.json",
        "**/items_ins_v2_1000.json",
        "**/items*.json",
    ):
        items = sorted(root.glob(pattern))
        if items:
            return items[0]
    return None


def webshop_rows(root):
    root = Path(root)
    item_file = find_webshop_items(root)
    if item_file is None:
        print("[WARN] WebShop item file not found; WebShop omitted")
        return []

    train_file = root / "train.jsonl"
    count = 6410
    if train_file.exists():
        try:
            count = min(6410, sum(1 for _ in train_file.open("rb")))
        except Exception:
            pass

    return [{
        "messages": [{"role": "user", "content": "<environment task>"}],
        "env_config": {
            "name": "common_agent",
            "kind": "webshop",
            "file_path": str(item_file),
            "session": idx,
            "num_products": 1000,
        },
    } for idx in range(count)]


def swe_rows(root):
    path = Path(root) / "train_2438.parquet"
    if not path.exists():
        return []

    table = pq.read_table(path)
    rows = []
    for x in table.to_pylist():
        problem = x.get("problem_statement") or x.get("problem") or ""
        if not problem:
            continue
        prompt = (
            f"Repository: {x.get('repo', '')}\n"
            f"Base commit: {x.get('base_commit', '')}\n\n"
            f"Issue:\n{problem}\n\n"
            "Produce the concrete code change or patch you would make. "
            "Include the exact verification/tests you would run."
        )
        rows.append({
            "messages": [{"role": "user", "content": "<repository task>"}],
            "env_config": {
                "name": "common_agent",
                "kind": "prompt_only",
                "prompt": prompt,
                "system": (
                    "You are a repository agent. Diagnose the issue, propose the exact "
                    "code change, and verify it carefully."
                ),
            },
        })
    return rows


def interleave(rows):
    buckets = {}
    for row in rows:
        key = row["env_config"]["kind"]
        buckets.setdefault(key, []).append(row)
    mixed = []
    while any(buckets.values()):
        for key in sorted(buckets):
            if buckets[key]:
                mixed.append(buckets[key].pop(0))
    return mixed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, choices=[1, 2], required=True)
    ap.add_argument("--v1-root", required=True)
    ap.add_argument("--v2-root", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    if args.stage == 1:
        root = Path(args.v1_root)
        rows = (
            awm_rows(root / "AgentWorldModel-1K")
            + alfworld_rows(root / "ALFWorld")
            + webshop_rows(root / "WebShop")
        )
    else:
        root = Path(args.v2_root)
        rows = (
            awm_rows(root / "AgentWorldModel-1K")
            + alfworld_rows(root / "ALFWorld")
            + webshop_rows(root / "WebShop")
            + swe_rows(root / "SWE-Gym")
        )

    write_jsonl(interleave(rows), args.output)


if __name__ == "__main__":
    main()
