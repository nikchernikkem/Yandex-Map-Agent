# Yandex-Map-Agent
[![Python](https://img.shields.io/badge/Python-%3E%3D3.12-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-%3E%3D0.3-green.svg)](https://python.langchain.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-%3E%3D0.1-green.svg)](https://python.langgraph.com/)


Проект по созданию LLM‑агента, который оценивает релевантность организаций на Яндекс.Картах широким рубричным запросам (например, «ресторан с верандой» или «романтичный джаз‑бар»). Агент должен сам находить необходимые данные и принимать решение о релевантности.

Данные предоставлены Яндексом и содержат ассессорскую разметку релевантности.

**Что внутри**
1. Агент на LangGraph с LLM через OpenAI‑compatible API.
1. Инструмент поиска Tavily.
1. FAISS‑ретривер примеров на эмбеддингах `ai-forever/FRIDA`.
1. Базовый скрипт TF‑IDF + CatBoost.


**Данные и постановка**
- Цель: предсказать `relevance` по всем остальным полям организации.
- Метки: `1.0` = `RELEVANT_PLUS`, `0.1` = `RELEVANT_MINUS`, `0.0` = `IRRELEVANT`.
- Метрика: `accuracy` (классы сбалансированы).
- Eval‑множество: 500 объектов, без класса `RELEVANT_MINUS` для простоты.
- Важно: не использовать eval для калибровки/подбора промпта или параметров. Это делается только на train.

**Материалы**
- Статья: https://habr.com/ru/articles/864646/
- Видео: https://www.youtube.com/watch?v=U6LbW2IFUQw
- Данные: https://disk.yandex.ru/d/6d5hFHvpAZjQdw

**Структура проекта**
- `src/agent.py` — граф LangGraph, сборка цепочки, запуск LLM‑агента.
- `src/tools.py` — инструменты агента (Tavily Search и FAISS‑retriever).
- `src/faiss_retriever.py` — загрузка индекса, поиск похожих примеров, форматирование выдачи.
- `src/scripts/baseline_tfidf_catboost.py` — baseline: TF‑IDF + CatBoost, обучение и валидация.
- `src/scripts/make_faiss_index.py` — построение FAISS‑индекса и метаданных (режимы split/full).
- `src/scripts/run_prediction.py` — запуск экспериментов агента, логирование предсказаний в `logs/agent_runs/`.

<br>
<figure>
  <p align="center">
    <img src="images/Flow.png" width="370" />
  </p>
  <figcaption align="center">Структура агента</figcaption>
</figure>


**Зачем агенту RAG‑тулза (FAISS-ретривер)**
- Чтобы в сложных или пограничных случаях быстро подтянуть похожие размеченные примеры из трейна и использовать их как калибровку решения. Это снижает риск «галлюцинаций», дает более стабильные решения на неочевидных запросах и помогает придерживаться обученной логики меток. По факту это few‑shot.
- Индекс строится по паре `QUERY + org_text`, где `org_text` включает `CITY` (первый компонент `address`), `normalized_main_rubric_name_ru`, `name`.


**Переменные окружения**
- `OPENAI_BASE_URL` — base URL для OpenAI‑compatible API.
- `OPENAI_API_KEY` — ключ для модели.
- `TAVILY_API_KEY` — ключ для Tavily Search.

**Что было сделано**
1. Разделяю трейновый датасет: 98% — для построения FAISS‑индекса (RAG), 2% — для настройки агента (tune_split).
2. Через GPT‑5.2 thinking выбираю 50 неоднозначных примеров из tune_split и прогоняю 3 стратегии:
- без тулзов (`system_prompt_base.jinja`),
- с редким использованием тулзов (`system_prompt_tools.jinja`),
- с частым использованием тулзов (`system_prompt_tools_2.jinja`).
3. Выбираю лучшую стратегию, строю FAISS‑индекс по 100% трейна и запускаю финальный эксперимент на валидационном сете.
4. Параллельно считаю бейзлайн: baseline_tfidf_catboost.py (baseline.ipynb) на валидации.

**Данные и артефакты**
- `data/raw/data_final_for_dls_new.jsonl` и `data/raw/data_final_for_dls_eval_new.jsonl` — исходные данные.
- `data/raw/data_final_for_dls_eval_new.jsonl` — валидация для экспериментов и baseline.

- `data/artifacts/faiss.index` и `data/artifacts/faiss_meta.json` — индекс и метаданные для retriever.
- `data/artifacts/faiss_split/train_split.jsonl` и `data/artifacts/faiss_split/tune_split.jsonl` — train/tune (98/2) для настройки агента.
- `data/artifacts/faiss_split/tune_split_hard_examples.jsonl`— 50 сложных примеров из tune для тестирования гипотез агента.

- `data/raw/data_final_for_dls_new.jsonl` — 100% трейна для финального индекса.
- `logs/agent_runs/` — логи предсказаний run_prediction.py.


**Скрипты**
Запуск baseline:

```bash
python src/scripts/baseline_tfidf_catboost.py --train-path data/raw/data_final_for_dls_new.jsonl --val-path data/artifacts/val.jsonl
```

Построение FAISS индекса (98/2 для настройки агента):

```bash
python src/scripts/make_faiss_index.py --artifacts-dir data/artifacts/faiss_split --mode split --split-frac 0.98
```

Построение FAISS индекса (100% трейна):

```bash
python src/scripts/make_faiss_index.py --artifacts-dir data/artifacts/faiss_full --mode full
```

Запуск эксперимента агента:

```bash
export FAISS_INDEX_DIR=data/artifacts/faiss_split  
python src/scripts/run_prediction.py --experiment no_tools --sample-size 50
```

Полный прогон на всем val:

```bash
export FAISS_INDEX_DIR=data/artifacts/faiss_full
python src/scripts/run_prediction.py --experiment web_search_rag --use-full
```

**Результаты экспериментов**
1. Baseline (TF-IDF + CatBoost, `baseline.ipynb`): val size = 500, accuracy = 0.8700, precision = 0.9013, recall = 0.8927, F1 = 0.8970.


