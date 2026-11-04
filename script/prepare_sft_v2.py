"""Rebuild the v2 candidate from immutable v1 raw data and recorded audit decisions.

Run from any directory: python3 script/prepare_sft_v2.py
Only the 48 sampled records were substantively audited. The other records retain
audit_status=not_reviewed. This distinction is retained in every output row.
"""

import difflib
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "data/audit_v2"
OUTPUT = ROOT / "data/sft_v2"
SEED = 20261104


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def normalized(text):
    return re.sub(r"[^\w]+", " ", text.casefold().replace("ё", "е")).strip()


def build():
    rules = json.loads((AUDIT / "scenario_rules.json").read_text())
    decisions = {r["sample_id"]: r for r in read_jsonl(AUDIT / "decisions.jsonl")}
    sample = {r["sample_id"]: r for r in read_jsonl(AUDIT / "audit_sample.jsonl")}
    assert len(decisions) == len(sample) == 48 and decisions.keys() == sample.keys()
    canonical_system = read_jsonl(ROOT / "data/sft_train.jsonl")[0]["messages"][0]["content"]
    rows = []
    source_hashes = {}
    for path in sorted((ROOT / "data/raw").glob("*.jsonl")):
        source_hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        for line_number, row in enumerate(read_jsonl(path), 1):
            sample_id = f"{path.stem}:{line_number:03d}"
            messages = row["messages"]
            assert [m["role"] for m in messages] == ["system", "user", "assistant"]
            if sample_id in decisions:
                assert messages == sample[sample_id]["messages"], f"Audit source changed: {sample_id}"
                assert messages[2]["content"] == decisions[sample_id]["original_answer"]
                messages[2]["content"] = decisions[sample_id]["revised_answer"]
            messages[0]["content"] = canonical_system
            scenario = next((r["scenario"] for r in rules["rules"]
                             if re.search(r["pattern"], messages[1]["content"], re.I)),
                            "other_" + path.stem)
            rows.append(dict(sample_id=sample_id, category=path.stem, messages=messages,
                             scenario=scenario,
                             audit_status=decisions.get(sample_id, {}).get("action", "not_reviewed")))
    assert len(rows) == 650

    # Union scenario families across all source categories. A shared group never
    # occurs in both splits. Similar question strings can merge entire families.
    parents = {r["scenario"]: r["scenario"] for r in rows}

    def find(key):
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    def union(left, right):
        a, b = sorted((find(left), find(right)))
        parents[b] = a

    texts = [normalized(r["messages"][1]["content"]) for r in rows]
    near_pairs = []
    for i, left in enumerate(rows):
        for j in range(i):
            matcher = difflib.SequenceMatcher(None, texts[i], texts[j], autojunk=False)
            if matcher.quick_ratio() < rules["fuzzy_threshold"]:
                continue
            similarity = matcher.ratio()
            if similarity >= rules["fuzzy_threshold"]:
                union(left["scenario"], rows[j]["scenario"])
                near_pairs.append(dict(left=left["sample_id"], right=rows[j]["sample_id"],
                                       similarity=round(similarity, 6)))
    for row in rows:
        row["scenario_group"] = find(row["scenario"])

    groups = sorted({r["scenario_group"] for r in rows})
    categories = sorted({r["category"] for r in rows})
    totals = Counter(r["category"] for r in rows)
    group_counts = {g: Counter(r["category"] for r in rows if r["scenario_group"] == g) for g in groups}
    for category in categories:
        assert sum(bool(group_counts[g][category]) for g in groups) >= 2, (
            f"Category {category} has only one group; review grouping before splitting")

    # Deterministic search for a group holdout near 15%, with every category in
    # both splits. The fraction can differ per category because groups are atomic.
    rng = random.Random(SEED)
    best = None
    for _ in range(20000):
        validation_groups = {g for g in groups if rng.random() < 0.2}
        counts = Counter()
        for group in validation_groups:
            counts.update(group_counts[group])
        if not all(1 <= counts[c] < totals[c] for c in categories):
            continue
        size = sum(counts.values())
        if not 50 <= size <= 180:
            continue
        cost = abs(size / len(rows) - 0.15) + sum(abs(counts[c] / totals[c] - 0.15) for c in categories) / len(categories)
        candidate = (cost, tuple(sorted(validation_groups)))
        if best is None or candidate < best:
            best = candidate
    assert best is not None, "No balanced group split found; review rules"
    validation_groups = set(best[1])
    splits = {
        "train": [r for r in rows if r["scenario_group"] not in validation_groups],
        "val": [r for r in rows if r["scenario_group"] in validation_groups],
    }
    for name, selected in splits.items():
        random.Random(SEED + (name == "val")).shuffle(selected)
    assert not ({r["scenario_group"] for r in splits["train"]} &
                {r["scenario_group"] for r in splits["val"]})
    assert not ({normalized(r["messages"][1]["content"]) for r in splits["train"]} &
                {normalized(r["messages"][1]["content"]) for r in splits["val"]})

    OUTPUT.mkdir(parents=True, exist_ok=True)
    output_hashes = {}
    for name, selected in splits.items():
        path = OUTPUT / f"sft_{name}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in selected))
        output_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    report = dict(
        version="sft_v2_candidate", seed=SEED, records=len(rows),
        audit_records=len(decisions), audit_actions=dict(Counter(d["action"] for d in decisions.values())),
        unaudited_records=sum(r["audit_status"] == "not_reviewed" for r in rows),
        source_hashes=source_hashes,
        audit_decisions_sha256=hashlib.sha256((AUDIT / "decisions.jsonl").read_bytes()).hexdigest(),
        scenario_rules_sha256=hashlib.sha256((AUDIT / "scenario_rules.json").read_bytes()).hexdigest(),
        scenario_group_count=len(groups), validation_groups=sorted(validation_groups),
        counts={name: dict(Counter(r["category"] for r in selected)) for name, selected in splits.items()},
        sizes={name: len(selected) for name, selected in splits.items()},
        output_hashes=output_hashes,
        limitations=["602 records have no new substantive review",
                     "Grouping uses explicit heuristic families, not expert semantic annotation",
                     "Near duplicates below the similarity threshold can remain",
                     "Sources checked for selected safety principles, not full product/legal certification",
                     "Changed data, split and training settings form a bundled experiment, not an ablation"],
    )
    (OUTPUT / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (OUTPUT / "group_assignments.jsonl").write_text("".join(json.dumps(
        dict(sample_id=r["sample_id"], category=r["category"], scenario=r["scenario"],
             scenario_group=r["scenario_group"], split=name, audit_status=r["audit_status"]), ensure_ascii=False) + "\n"
        for name, selected in splits.items() for r in selected))
    (OUTPUT / "similar_question_pairs.jsonl").write_text("".join(json.dumps(p) + "\n" for p in near_pairs))
    print(json.dumps({k: report[k] for k in ("sizes", "counts", "audit_actions", "unaudited_records", "scenario_group_count")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    build()
