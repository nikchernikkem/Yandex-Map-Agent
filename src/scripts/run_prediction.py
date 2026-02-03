import re
import json
import sys
from pathlib import Path

import pandas as pd

from langchain_core.messages import HumanMessage, SystemMessage

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent import graph


# ----------------------------
# Load val_data
# ----------------------------
val_data = pd.read_json(r"data\\artifacts\\val.jsonl", lines=True)

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

def predict_one(row: pd.Series) -> dict:
    payload = row_to_agent_input(row)
    prompt = json.dumps(payload, ensure_ascii=False)

    out_state = graph.invoke({"messages": [HumanMessage(content=prompt)]})
    final_msg = out_state["messages"][-1]  # финальный ответ агента (после tool-циклов)

    parsed = extract_json_from_text(final_msg.content if hasattr(final_msg, "content") else "")
    return {
        "pred_raw": final_msg.content if hasattr(final_msg, "content") else None,
        "pred_parsed": parsed,
    }

# ----------------------------
# 6) Quick test on a small sample (чтобы не сжечь бюджет)

def eval_on_sample(df: pd.DataFrame, n: int = 20):
    # sample = df.sample(n=n, random_state=42).reset_index(drop=True)
    sample = val_data.iloc[n,:]

    y_true = []
    y_pred = []
    bad = 0

    # for i, row in sample.iterrows():
    #     true_label = float(row["relevance_new"])
    #     res = predict_one(row)
    #     parsed = res["pred_parsed"]

    #     if not parsed or "relevance" not in parsed:
    #         bad += 1
    #         pred_label = None
    #     else:
    #         pred_label = float(parsed["relevance"])

    #     y_true.append(true_label)
    #     y_pred.append(pred_label)

    #     print(f"[{i}] true={true_label} pred={pred_label}")

    true_label = float(sample["relevance_new"])
    res = predict_one(sample)
    parsed = res["pred_parsed"]

    if not parsed or "relevance" not in parsed:
        bad += 1
        pred_label = None
    else:
        pred_label = float(parsed["relevance"])

    y_true.append(true_label)
    y_pred.append(pred_label)

    print(f"true={true_label} pred={pred_label}")


    # считаем accuracy только по тем, где удалось распарсить ответ
    pairs = [(t, p) for t, p in zip(y_true, y_pred) if p is not None]
    acc = sum(1 for t, p in pairs if t == p) / len(pairs) if pairs else 0.0

    print(f"\nParsed OK: {len(pairs)}/{n}, bad_parse={bad}, accuracy_on_parsed={acc:.4f}")




eval_on_sample(val_data, n=0)
