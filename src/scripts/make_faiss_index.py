import json
from pathlib import Path
import faiss
import pandas as pd
from src.faiss_retriever import FaissExampleRetriever

#  train_data
train_data = pd.read_json(r"data\raw\data_final_for_dls_new.jsonl", lines=True)

#  train/val (90/10)
train_frac = 0.9
train_data = train_data.sample(frac=1.0, random_state=42).reset_index(drop=True)
split_idx = int(len(train_data) * train_frac)
train_df = train_data.iloc[:split_idx].reset_index(drop=True)
val_df = train_data.iloc[split_idx:].reset_index(drop=True)


retriever = FaissExampleRetriever(embed_model_name="ai-forever/FRIDA", label_col="relevance_new").build(train_df)

#  FAISS + meta 
artifacts_dir = Path("data") / "artifacts"
artifacts_dir.mkdir(parents=True, exist_ok=True)

index_path = artifacts_dir / "faiss.index"
meta_path = artifacts_dir / "faiss_meta.json"
val_path = artifacts_dir / "val.jsonl"

faiss.write_index(retriever.index, str(index_path))
with meta_path.open("w", encoding="utf-8") as f:
    json.dump(retriever.meta, f, ensure_ascii=False)

val_df.to_json(val_path, orient="records", lines=True, force_ascii=False)
