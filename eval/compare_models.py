"""Compare Text-to-SQL benchmark results across models and prompt conditions.

Run from the project root (LM Studio can be OFF), for example:
    python eval/compare_models.py ^
        --model "Qwen 2.5 7B=eval/results/qwen_7b_results.csv" ^
        --model "Qwen3-VL 4B=eval/results/qwen3vl_4b_results.csv"

(in PowerShell, put it on one line instead of using ^)

Writes tables and a figure to eval/results/analysis_comparison/.
Raw result files are never modified.
"""
import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ("schema_only", "with_semantic_layer")
DIFFICULTIES = ("easy", "medium", "hard")


def error_subtype(row):
    """Same categories as analyze_qwen_results.py."""
    status = row["status"]
    error = str(row["error"]).lower()
    if status == "execution_error":
        if "exceeded 12 seconds" in error or "timed out" in error:
            return "timeout"
        if "binderexception" in error or "binder error" in error:
            return "sql_binder_error"
        if "parserexception" in error or "parser error" in error:
            return "sql_parser_error"
        return "other_execution_error"
    return status


def wilson(k, n, z=1.96):
    """95% Wilson interval for a proportion, in percent."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (round(100 * (centre - half), 1), round(100 * (centre + half), 1))


def load(label, path):
    df = pd.read_csv(path, keep_default_na=False, encoding="utf-8-sig")
    need = {"id", "difficulty", "condition", "status", "error"}
    if not need.issubset(df.columns):
        raise SystemExit(f"{path} is missing columns: {sorted(need - set(df.columns))}")
    if df.duplicated(["id", "condition"]).any():
        raise SystemExit(f"{path} has duplicate (id, condition) rows")
    df = df.copy()
    df["model_label"] = label
    df["error_subtype"] = df.apply(error_subtype, axis=1)
    df["is_correct"] = df["status"] == "correct"
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True, help='"Label=path/to/results.csv" (repeat per model)')
    ap.add_argument("--output-dir", type=Path, default=ROOT / "eval" / "results" / "analysis_comparison")
    args = ap.parse_args()

    frames = []
    for item in args.model:
        if "=" not in item:
            raise SystemExit(f'Use "Label=path" for --model, got: {item}')
        label, path = item.split("=", 1)
        frames.append(load(label.strip(), Path(path.strip())))
    data = pd.concat(frames, ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1) overall accuracy, with 95% intervals, per model and condition
    rows = []
    for (label, cond), g in data.groupby(["model_label", "condition"], sort=False):
        n, k = len(g), int(g["is_correct"].sum())
        lo, hi = wilson(k, n)
        row = {"model": label, "condition": cond, "n": n, "correct": k,
               "accuracy_pct": round(100 * k / n, 1), "ci95_low": lo, "ci95_high": hi}
        for d in DIFFICULTIES:
            gd = g[g["difficulty"] == d]
            row[f"{d}_n"] = len(gd)
            row[f"{d}_pct"] = round(100 * gd["is_correct"].mean(), 1) if len(gd) else None
        counts = g.loc[~g["is_correct"], "error_subtype"].value_counts()
        for kind in ("timeout", "sql_binder_error", "sql_parser_error", "other_execution_error",
                     "wrong_result", "blocked_sql", "invalid_sql", "llm_error"):
            row[kind] = int(counts.get(kind, 0))
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(args.output_dir / "model_comparison.csv", index=False)

    # 2) paired comparison of the two conditions within each model
    paired = []
    for label, g in data.groupby("model_label", sort=False):
        w = g.pivot(index="id", columns="condition", values="is_correct")
        if not set(CONDITIONS).issubset(w.columns):
            continue
        a, b = w["schema_only"], w["with_semantic_layer"]
        paired.append({"model": label, "both_correct": int((a & b).sum()),
                       "only_schema_only": int((a & ~b).sum()),
                       "only_with_semantic_layer": int((~a & b).sum()),
                       "neither": int((~a & ~b).sum())})
    paired = pd.DataFrame(paired)
    paired.to_csv(args.output_dir / "paired_conditions.csv", index=False)

    # 3) paired comparison of the first two models within each condition
    labels = list(dict.fromkeys(data["model_label"]))
    cross = pd.DataFrame()
    if len(labels) >= 2:
        m1, m2 = labels[0], labels[1]
        out = []
        for cond in CONDITIONS:
            w = data[data["condition"] == cond].pivot(index="id", columns="model_label", values="is_correct")
            if {m1, m2}.issubset(w.columns):
                a, b = w[m1], w[m2]
                out.append({"condition": cond, "both_correct": int((a & b).sum()),
                            f"only_{m1}": int((a & ~b).sum()), f"only_{m2}": int((~a & b).sum()),
                            "neither": int((~a & ~b).sum())})
        cross = pd.DataFrame(out)
        cross.to_csv(args.output_dir / "paired_models.csv", index=False)

    # figure: accuracy by difficulty for every model and condition
    fig, ax = plt.subplots(figsize=(9, 4.8))
    groups = [(r["model"], r["condition"]) for _, r in summary.iterrows()]
    width = 0.8 / max(1, len(groups))
    for i, (model, cond) in enumerate(groups):
        r = summary[(summary["model"] == model) & (summary["condition"] == cond)].iloc[0]
        vals = [r[f"{d}_pct"] or 0 for d in DIFFICULTIES] + [r["accuracy_pct"]]
        xs = [j + (i - (len(groups) - 1) / 2) * width for j in range(4)]
        bars = ax.bar(xs, vals, width, label=f"{model}, {cond.replace('_', ' ')}")
        for rect, v in zip(bars, vals):
            ax.annotate(f"{v:.0f}", (rect.get_x() + rect.get_width() / 2, v), xytext=(0, 2),
                        textcoords="offset points", ha="center", fontsize=8)
    ax.set_xticks(range(4)); ax.set_xticklabels(["easy", "medium", "hard", "overall"])
    ax.set_ylim(0, 105); ax.set_ylabel("Execution accuracy (%)")
    ax.set_title("Text-to-SQL accuracy by difficulty")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(args.output_dir / "accuracy_by_difficulty.png", dpi=170)
    plt.close(fig)

    pd.set_option("display.width", 200)
    print("\nOVERALL AND BY DIFFICULTY")
    print(summary[["model", "condition", "n", "correct", "accuracy_pct", "ci95_low", "ci95_high",
                   "easy_pct", "medium_pct", "hard_pct"]].to_string(index=False))
    print("\nFAILURE CATEGORIES (count of unsuccessful evaluations)")
    print(summary[["model", "condition", "timeout", "sql_binder_error", "sql_parser_error",
                   "other_execution_error", "wrong_result", "blocked_sql", "invalid_sql",
                   "llm_error"]].to_string(index=False))
    print("\nPAIRED: schema-only vs semantic layer, within each model")
    print(paired.to_string(index=False))
    if not cross.empty:
        print("\nPAIRED: first two models, within each condition")
        print(cross.to_string(index=False))
    print("\nSaved to:", args.output_dir)
    print("Note: with 50 questions the 95% intervals are wide; differences of a few points are not meaningful.")


if __name__ == "__main__":
    main()
