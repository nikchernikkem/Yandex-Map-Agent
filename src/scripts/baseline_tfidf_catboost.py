import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


TEXT_COLUMNS = [
    "Text",
    "name",
    "address",
    "normalized_main_rubric_name_ru",
    "prices_summarized",
    "reviews_summarized",
]


def build_text(df: pd.DataFrame) -> pd.Series:
    safe = df.reindex(columns=TEXT_COLUMNS).fillna("")
    return safe.astype(str).agg(" ".join, axis=1)


def resolve_label(df: pd.DataFrame, preferred: str) -> pd.Series:
    if preferred in df.columns:
        return df[preferred]
    if "relevance_new" in df.columns:
        return df["relevance_new"]
    if "relevance" in df.columns:
        return df["relevance"]
    raise KeyError("No label column found (expected relevance_new or relevance).")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TF-IDF + CatBoost baseline.")
    parser.add_argument(
        "--train-path",
        type=Path,
        default=Path("data/raw/data_final_for_dls_new.jsonl"),
        help="Path to train jsonl.",
    )
    parser.add_argument(
        "--val-path",
        type=Path,
        default=Path("data/artifacts/val.jsonl"),
        help="Path to validation jsonl.",
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default="relevance_new",
        help="Label column name (fallback: relevance_new -> relevance).",
    )
    parser.add_argument("--max-features", type=int, default=120000)
    parser.add_argument("--min-df", type=int, default=3)
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--l2-leaf-reg", type=float, default=3.0)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)

    parser.add_argument("--task-type", type=str, default="GPU")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_df = pd.read_json(args.train_path, lines=True)
    val_df = pd.read_json(args.val_path, lines=True)

    y_train = resolve_label(train_df, args.label_col).astype(int)
    y_val = resolve_label(val_df, args.label_col).astype(int)

    train_text = build_text(train_df)
    val_text = build_text(val_df)

    vectorizer = TfidfVectorizer(
        max_features=args.max_features,
        min_df=args.min_df,
        ngram_range=(1, args.ngram_max),
        lowercase=True,
        token_pattern=r"(?u)\b\w+\b",
    )

    x_train = vectorizer.fit_transform(train_text)
    x_val = vectorizer.transform(val_text)

    train_pool = Pool(x_train, y_train)
    val_pool = Pool(x_val, y_val)

    model = CatBoostClassifier(
        iterations=args.iterations,
        depth=args.depth,
        learning_rate=args.learning_rate,
        l2_leaf_reg=args.l2_leaf_reg,
        loss_function="Logloss",
        eval_metric="F1",
        random_seed=args.random_seed,
        verbose=100,
        allow_writing_files=False,
        task_type=args.task_type,
        od_type="Iter",
        od_wait=150,
    )

    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    val_proba = model.predict_proba(x_val)[:, 1]
    y_pred = (val_proba >= args.threshold).astype(int)

    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)

    print(f"Validation size: {len(y_val)}")
    print(f"Accuracy: {acc:.4f}")
    print(f"precision: {precision:.4f}")
    print(f"recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")


if __name__ == "__main__":
    main()
