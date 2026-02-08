import json
import os
from pathlib import Path

import faiss
from langchain_community.tools import TavilySearchResults
from langchain_core.tools import tool

from src.faiss_retriever import FaissExampleRetriever

from dotenv import load_dotenv
load_dotenv()

# ----------------------------
# Tools
# ----------------------------

# Tavily для поиска в интернете - готовый Tool-объект
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
search_tool = TavilySearchResults(
    max_results=10,
    include_answer=True,
    include_raw_content=False,
    tavily_api_key=TAVILY_API_KEY,
)

# Retriever для few-shot

INDEX_DIR = os.getenv("FAISS_INDEX_DIR")
ARTIFACTS_DIR = Path(INDEX_DIR) if INDEX_DIR else Path("data") / "artifacts" / "faiss_split"
INDEX_PATH = ARTIFACTS_DIR / "faiss.index"
META_PATH = ARTIFACTS_DIR / "faiss_meta.json"

retriever = FaissExampleRetriever(
    embed_model_name="ai-forever/FRIDA",
    label_col="relevance_new",
)
retriever.index = faiss.read_index(str(INDEX_PATH))
with META_PATH.open("r", encoding="utf-8") as f:
    retriever.meta = json.load(f)

def make_retrieve_tool(retriever: FaissExampleRetriever):
    # Декоратор превращает функцию в Tool с именем, схемой аргументов и описанием
    @tool
    def retrieve_similar_examples(query: str, org_text: str, k: int = 5) -> str:
        """
        Find top-k similar labeled TRAIN examples for the given (query, org_text) pair.
        Returns a JSON string list of examples with scores and labels.
        """
        examples = retriever.retrieve_similar_examples(query=query, org_text=org_text, k=k)
        return json.dumps(examples, ensure_ascii=False)

    return retrieve_similar_examples

# Пример использования без агента
# row = train_data.iloc[0]
# org_text = make_org_text_from_row(row)
# print(retriever.retrieve_similar_examples(row["Text"], org_text, k=5))

rag_tool = make_retrieve_tool(retriever)
tools = [search_tool, rag_tool]
