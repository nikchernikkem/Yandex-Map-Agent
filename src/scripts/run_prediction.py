import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
from langchain_core.messages import HumanMessage
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent import build_graph
from src.tools import search_tool, rag_tool


"""
Запуск одного эксперимента агента с выбранной конфигурацией тулзов на подмножестве валидации,
и опционально прогон этой же конфигурации на всем подмножестве.

Использование:
- Выберите ровно один эксперимент через --experiment (no_tools, web_search, rag, web_search_rag).
- Оценка на --sample-size (по умолчанию 50), опционально на --full-size (по умолчанию 500).

Выход:
- Печатает метрики только по распарсенным предиктам (accuracy, macro F1, precision, recall).
- Сохраняет построчные предсказания в data/artifacts/agent_runs, если задан --output-dir.

Ожидаемый вход:
- JSONL валидации с полями: Text, name, address, normalized_main_rubric_name_ru,
  prices_summarized, reviews_summarized, relevance_new.

Примечания:
- Метки бинарные (0/1). Любое неточное значение считается ошибкой парсинга.
- Использование тулзов определяется выбранным экспериментом.
"""


ALLOWED_LABELS = [0.0, 1.0]
EXPERIMENTS = {
    "no_tools": [],
    "web_search": [search_tool],
    "rag": [rag_tool],
    "web_search_rag": [search_tool, rag_tool],
}
PROMPT_FILES = {
    "no_tools": PROJECT_ROOT / "src" / "prompts" / "system_prompt_base.jinja",
    "web_search": PROJECT_ROOT / "src" / "prompts" / "system_prompt_web.jinja",
    "rag": PROJECT_ROOT / "src" / "prompts" / "system_prompt_rag.jinja",
    "web_search_rag": PROJECT_ROOT / "src" / "prompts" / "system_prompt_tools.jinja",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Agent experiments with tool configs.")
    parser.add_argument(
        "--val-path",
        type=Path,
        default=Path("data/artifacts/val.jsonl"),
        help="Path to validation jsonl.",
    )
    parser.add_argument("--sample-size", type=int, default=50, help="Sample size for experiment.")
    parser.add_argument("--use-full", action="store_true", help="Using full val df for final run.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument(
        "--experiment",
        type=str,
        choices=sorted(EXPERIMENTS.keys()),
        default="no_tools",
        help="Which experiment to run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/artifacts/agent_runs"),
        help="Directory to store predictions.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print progress every N samples (0 to disable).",
    )
    return parser.parse_args()


# ----------------------------
# Helpers: row -> agent input, parse agent output
# ----------------------------
def row_to_agent_input(row: pd.Series) -> dict:
    # Передаём агенту только фичи (без relevance!)
    return {
        "Text": row.get("Text"),
        "name": row.get("name"),
        "address": row.get("address"),
        "normalized_main_rubric_name_ru": row.get("normalized_main_rubric_name_ru"),
        "prices_summarized": row.get("prices_summarized"),
        "reviews_summarized": row.get("reviews_summarized"),
        # "permalink": row.get("permalink"),
    }


def extract_json_from_text(text: str) -> dict | None:
    if not text:
        return None
    # на случай, если модель добавила лишний текст вокруг JSON
    # убираем ```json и ```
    cleaned = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).replace("```", "").strip()

    # находим первый JSON-объект
    m = re.search(r"\{.*\}", cleaned, flags=re.S)
    if not m:
        return None

    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def coerce_label(value):
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    for label in ALLOWED_LABELS:
        if abs(v - label) <= 1e-6:
            return label
    return None


def predict_one(row: pd.Series, graph):
    payload = row_to_agent_input(row)
    prompt = json.dumps(payload, ensure_ascii=False)

    out_state = graph.invoke({"messages": [HumanMessage(content=prompt)]})
    final_msg = out_state["messages"][-1]
    raw = final_msg.content if hasattr(final_msg, "content") else None

    parsed = extract_json_from_text(raw if raw else "")
    pred_label = coerce_label(parsed.get("relevance")) if parsed else None

    return {
        "pred_raw": raw,
        "pred_parsed": parsed,
        "pred_label": pred_label,
    }


def select_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n <= 0 or n >= len(df):
        return df.copy()
    return df.sample(n=n, random_state=seed).copy()


def evaluate_df(df: pd.DataFrame, graph, progress_every: int = 25):
    y_true = []
    y_pred = []
    rows = []
    bad_parse = 0

    for i, (idx, row) in enumerate(df.iterrows(), start=1):
        true_label = float(row["relevance_new"])
        res = predict_one(row, graph)

        if res["pred_label"] is None:
            bad_parse += 1
        else:
            y_true.append(true_label)
            y_pred.append(res["pred_label"])

        rows.append(
            {
                "row_id": int(idx),
                "true_label": true_label,
                "pred_label": res["pred_label"],
                "pred_raw": res["pred_raw"],
                "pred_parsed": res["pred_parsed"],
            }
        )

        if progress_every and i % progress_every == 0:
            print(f"Processed {i}/{len(df)}")

    parsed_count = len(y_pred)
    parsed_rate = parsed_count / len(df) if len(df) else 0.0

    # считаем accuracy только по тем, где удалось распарсить ответ
    if parsed_count:
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro", labels=ALLOWED_LABELS, zero_division=0)
        precision = precision_score(y_true, y_pred, average="macro", labels=ALLOWED_LABELS, zero_division=0)
        recall = recall_score(y_true, y_pred, average="macro", labels=ALLOWED_LABELS, zero_division=0)
    else:
        acc = f1 = precision = recall = 0.0

    metrics = {
        "total": len(df),
        "parsed_ok": parsed_count,
        "bad_parse": bad_parse,
        "parsed_rate": parsed_rate,
        "accuracy": acc,
        "f1_macro": f1,
        "precision_macro": precision,
        "recall_macro": recall,
    }

    return metrics, rows


def save_jsonl(rows, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def print_metrics(name: str, metrics: dict) -> None:
    print(
        f"{name}: total={metrics['total']} parsed={metrics['parsed_ok']} "
        f"bad={metrics['bad_parse']} parsed_rate={metrics['parsed_rate']:.3f} "
        f"acc={metrics['accuracy']:.4f} f1_macro={metrics['f1_macro']:.4f} "
        f"prec_macro={metrics['precision_macro']:.4f} recall_macro={metrics['recall_macro']:.4f}"
    )

def load_prompt(experiment: str) -> str:
    path = PROMPT_FILES[experiment]
    return path.read_text(encoding="utf-8")


def main() -> None:
    args = parse_args()

    # ----------------------------
    # Load val_data
    # ----------------------------
    val_df = pd.read_json(args.val_path, lines=True)
    val_df = val_df[val_df["relevance_new"] != 0.1]

    name = args.experiment
    tools = EXPERIMENTS[name]
    prompt = load_prompt(name)

    if args.use_full:
        full_df = val_df.copy()
        print(f"\nRunning {name} on full size: {len(full_df)}")
        graph = build_graph(tools, system_prompt=prompt)
        metrics, rows = evaluate_df(full_df, graph, progress_every=args.progress_every)
        print_metrics(f"{name}_full", metrics)
        if args.output_dir:
            save_jsonl(rows, args.output_dir / f"{name}_full.jsonl")

    else:
        sample_df = select_sample(val_df, args.sample_size, args.seed)

        print(f"Validation size: {len(val_df)}")
        print(f"Experiment sample size: {len(sample_df)}")

        print(f"\nRunning experiment: {name}")
        graph = build_graph(tools, system_prompt=prompt)
        metrics, rows = evaluate_df(sample_df, graph, progress_every=args.progress_every)
        print_metrics(name, metrics)

        if args.output_dir:
            save_jsonl(rows, args.output_dir / f"{name}_sample.jsonl")


if __name__ == "__main__":
    main()
