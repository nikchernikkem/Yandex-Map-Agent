import argparse
import json
import sys
from pathlib import Path

import faiss
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.faiss_retriever import FaissExampleRetriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FAISS index for RAG.")
    parser.add_argument(
        "--train-path",
        type=Path,
        default=Path(r"data\raw\data_final_for_dls_new.jsonl"),
        help="Path to full train jsonl.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("data") / "artifacts",
        help="Directory to store faiss index and splits.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["split", "full"],
        default="split",
        help="split: 98/2 for tuning; full: 100% train for final index.",
    )
    parser.add_argument("--split-frac", type=float, default=0.98, help="Train fraction for split mode.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    #  train_data
    train_data = pd.read_json(args.train_path, lines=True)

    #  train/val (98/2)
    train_df = None
    val_df = None
    if args.mode == "split":
        train_data = train_data.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
        split_idx = int(len(train_data) * args.split_frac)
        train_df = train_data.iloc[:split_idx].reset_index(drop=True)
        val_df = train_data.iloc[split_idx:].reset_index(drop=True)
    else:
        train_df = train_data.reset_index(drop=True)

    retriever = FaissExampleRetriever(
        embed_model_name="ai-forever/FRIDA",
        label_col="relevance_new",
    ).build(train_df)

    #  FAISS + meta 
    artifacts_dir = args.artifacts_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    index_path = artifacts_dir / "faiss.index"
    meta_path = artifacts_dir / "faiss_meta.json"

    faiss.write_index(retriever.index, str(index_path))
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(retriever.meta, f, ensure_ascii=False)

    if args.mode == "split":
        train_path = artifacts_dir / "train_split.jsonl"
        val_path = artifacts_dir / "tune_split.jsonl"
        train_df.to_json(train_path, orient="records", lines=True, force_ascii=False)
        val_df.to_json(val_path, orient="records", lines=True, force_ascii=False)
    # else:
    #     train_path = artifacts_dir / "train_full.jsonl"
    #     train_df.to_json(train_path, orient="records", lines=True, force_ascii=False)


if __name__ == "__main__":
    main()
