import numpy as np
import pandas as pd

# FAISS
import faiss

# Embeddings (локально, без API)
from sentence_transformers import SentenceTransformer


# -------------------------
# Утилиты подготовки текста
# -------------------------
def _truncate(s: str | None, n: int) -> str:
    if not s:
        return ""
    s = str(s).strip()
    return s if len(s) <= n else s[:n] + "..."

def _city_from_address(address: str | None) -> str:
    if not address:
        return ""
    return str(address).split(",")[0].strip()

def make_org_text_from_row(row: pd.Series,
                           max_reviews_chars: int = 800,
                           max_prices_chars: int = 400) -> str:
    """
    Делает компактный текст профиля организации.
    НЕ включает метку (relevance) — чтобы не было утечки.
    """
    name = _truncate(row.get("name"), 200)
    rubric = _truncate(row.get("normalized_main_rubric_name_ru"), 120)
    city = _city_from_address(row.get("address"))
    # prices = _truncate(row.get("prices_summarized"), max_prices_chars)
    # reviews = _truncate(row.get("reviews_summarized"), max_reviews_chars)

    return (
        f"CITY: {city}\n"
        f"RUBRIC: {rubric}\n"
        f"NAME: {name}\n"
        # f"PRICES: {prices}\n"
        # f"REVIEWS: {reviews}\n"
    ).strip()

def make_pair_text(query: str, org_text: str) -> str:
    """
    Это то, что мы эмбеддим для поиска похожих примеров.
    """
    q = (query or "").strip()
    return f"QUERY: {q}\n{org_text}".strip()


# -------------------------
# Ретривер на FAISS
# -------------------------
class FaissExampleRetriever:
    """
    Векторный ретривер train-примеров: ищет похожие (query + org_text) пары.
    """
    def __init__(self, embed_model_name: str = "ai-forever/FRIDA",
                 label_col: str = "relevance_new"):
        self.model = SentenceTransformer(embed_model_name, device="cuda")
        self.label_col = label_col

        self.index = None
        self.meta = None  # список метаданных по строкам train (без текстов простыней)

    def _encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        # normalize_embeddings=True => косинусная близость = inner product
        emb = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        return np.asarray(emb, dtype=np.float32)

    def build(self, train_df: pd.DataFrame,
              max_reviews_chars: int = 800,
              max_prices_chars: int = 400,
              batch_size: int = 64):
        """
        Строит FAISS индекс по train_df.
        """
        pair_texts = []
        meta = []

        for i, row in train_df.iterrows():
            q = row.get("Text")
            org_text = make_org_text_from_row(
                row,
                max_reviews_chars=max_reviews_chars,
                max_prices_chars=max_prices_chars,
            )
            pair_texts.append(make_pair_text(q, org_text))

            meta.append({
                "row_idx": int(i),
                "permalink": row.get("permalink"),
                "Text": row.get("Text"),
                "label": float(row.get(self.label_col)) if row.get(self.label_col) is not None else None,
                "rubric": row.get("normalized_main_rubric_name_ru"),
                "city": _city_from_address(row.get("address")),
                "name": row.get("name"),
            })

        vecs = self._encode(pair_texts, batch_size=batch_size)
        dim = vecs.shape[1]

        # Косинус: т.к. мы нормализовали эмбеддинги, используем IP (скалярное произведение)
        index = faiss.IndexFlatIP(dim)
        index.add(vecs)

        self.index = index
        self.meta = meta
        return self

    def retrieve_similar_examples(self, query: str, org_text: str, k: int = 5):
        """
        Возвращает top-k похожих train-примеров.
        """
        if self.index is None or self.meta is None:
            raise RuntimeError("FAISS индекс не построен. Сначала вызови retriever.build(train_df).")

        qtext = make_pair_text(query, org_text)
        qvec = self._encode([qtext], batch_size=1)

        scores, idxs = self.index.search(qvec, k)
        scores = scores[0].tolist()
        idxs = idxs[0].tolist()

        out = []
        for score, pos in zip(scores, idxs):
            if pos < 0:
                continue
            m = self.meta[pos]
            out.append({
                "score": float(score),
                "train_row_idx": m["row_idx"],
                "permalink": m["permalink"],
                "label": m["label"],
                "Text": m["Text"],
                "city": m["city"],
                "rubric": m["rubric"],
                "name": m["name"],
            })
        return out
