"""Запуск агента на eval-датасете."""
import os
import re
import json
import time
from tqdm.auto import tqdm
from langchain_core.messages import SystemMessage, HumanMessage

from config import (
    DATA_TRAIN_PATH, DATA_EVAL_PATH,
    AGENT_CACHE_DIR, VALID_LABELS, MAX_RETRIES, RECURSION_LIMIT
)
from utils import load_data, split_train_val, evaluate, get_embeddings
from agent.prompts import SYSTEM_PROMPT, build_task_message
from agent.tools import init_similarity_search
from agent.graph import build_graph


REFUSAL_MARKERS = (
    "не обладает собственным мнением",
    "разговоры на некоторые темы"
)


def parse_label(text: str) -> str:
    """Парсит метку из ответа модели."""
    try:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            lab = json.loads(m.group(0)).get("label", "").upper()
            if lab in VALID_LABELS:
                return lab
    except Exception:
        pass
    
    for lab in ["IRRELEVANT", "PARTIAL", "RELEVANT"]:
        if lab in text.upper():
            return lab
    return "IRRELEVANT"


def run_agent(app, llm, row, cache_dir):
    """Запускает агента для одной строки с кэшированием."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{row['permalink']}.json"
    
    if path.exists():
        with open(path) as f:
            return json.load(f)
    
    result = None
    for attempt in range(MAX_RETRIES):
        try:
            result = app.invoke({
                "messages": [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=build_task_message(row))
                ],
                "tool_calls_count": 0,
            }, config={"recursion_limit": RECURSION_LIMIT})
            break
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                print(f"Не удалось обработать {row['permalink']}: {e}")
                return {"label": "IRRELEVANT", "error": str(e), "trace": []}
            time.sleep(20 * (attempt + 1))
    
    final_text = result["messages"][-1].content
    
    # Обработка отказов цензуры GigaChat
    if any(m in final_text for m in REFUSAL_MARKERS):
        try:
            resp = llm.invoke(
                f"Классифицируй релевантность организации запросу. "
                f"Ответь одним словом: RELEVANT, PARTIAL или IRRELEVANT.\n"
                f"Запрос: {row['Text']}\n"
                f"Организация: {row['name']}, "
                f"рубрика: {row['normalized_main_rubric_name_ru']}"
            )
            final_text = resp.content
        except Exception as e:
            print(f"Fallback тоже упал: {e}")
    
    # Сохраняем полный трейс для анализа
    trace = []
    for m in result["messages"]:
        entry = {"role": m.type, "content": str(m.content)[:1500]}
        if getattr(m, "tool_calls", None):
            entry["tool_calls"] = [
                {"name": t["name"], "args": t["args"]}
                for t in m.tool_calls
            ]
        trace.append(entry)
    
    out = {
        "label": parse_label(final_text),
        "final": final_text,
        "trace": trace
    }
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    
    time.sleep(3)  # rate limit
    return out


def run_batch(app, llm, df, cache_dir):
    """Обрабатывает батч данных."""
    preds = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        preds.append(run_agent(app, llm, row, cache_dir)["label"])
    return preds


def main():
    print("Загрузка данных...")
    train, eval_df = load_data(DATA_TRAIN_PATH, DATA_EVAL_PATH)
    train_part, val_part = split_train_val(train)
    print(f"train_part: {len(train_part)}, val_part: {len(val_part)}")
    
    print("Инициализация similarity search...")
    core_emb = get_embeddings(train_part, cache_name="train_part_emb")
    init_similarity_search(train_part, core_emb)
    
    print("Построение графа агента...")
    app, llm = build_graph()
    
    print("Запуск на eval...")
    preds = run_batch(app, llm, eval_df, AGENT_CACHE_DIR / "eval")
    
    metrics = evaluate(eval_df["label"].tolist(), preds)
    print("\n=== Результаты ===")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
