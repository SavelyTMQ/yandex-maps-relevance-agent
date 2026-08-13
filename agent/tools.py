"""Инструменты агента: web_search, fetch_website, find_similar_examples."""
import re
import requests
import numpy as np
from bs4 import BeautifulSoup
from ddgs import DDGS
from langchain_core.tools import tool
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL, LABEL_MAP


# Глобальные переменные для find_similar_examples
# Инициализируются через init_similarity_search()
_emb_model = None
_core_df = None
_core_emb = None


def init_similarity_search(core_df, core_emb):
    """Инициализация модели и данных для поиска похожих примеров."""
    global _emb_model, _core_df, _core_emb
    _emb_model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    _emb_model.max_seq_length = 128
    _core_df = core_df
    _core_emb = core_emb


@tool
def web_search(query: str) -> str:
    """Поиск в интернете. Используй, чтобы проверить факты об организации:
    наличие атрибута (веранда, живая музыка), тип заведения, закрыта ли она."""
    try:
        results = DDGS().text(query, region="ru-ru", max_results=5)
        if not results:
            return "Ничего не найдено."
        return "\n\n".join(
            f"[{r['title']}]({r['href']})\n{r['body']}" for r in results
        )[:800]
    except Exception as e:
        return f"Ошибка поиска: {e}"


@tool
def fetch_website(url: str) -> str:
    """Загружает страницу по URL и возвращает её текст."""
    try:
        resp = requests.get(url, timeout=10, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        })
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = re.sub(r"\s+", " ", soup.get_text(separator=" ")).strip()
        return text[:1000] if text else "Страница пуста."
    except Exception as e:
        return f"Не удалось загрузить страницу: {e}"


@tool
def find_similar_examples(query_and_rubric: str) -> str:
    """Ищет в базе асессорской разметки похожие оценённые пары.
    Формат входа: 'запрос | рубрика организации'."""
    if _emb_model is None:
        return "Ошибка: similarity search не инициализирован."
    
    q = _emb_model.encode(
        [f"query: {query_and_rubric}"], normalize_embeddings=True
    )
    idx = np.argsort(-(_core_emb @ q.T).ravel())[:5]
    
    out = []
    for i in idx:
        r = _core_df.iloc[i]
        out.append(
            f"Запрос: {r['Text']} | Организация: {r['name']} "
            f"(рубрика: {r['normalized_main_rubric_name_ru']}) "
            f"→ оценка асессора: {LABEL_MAP[r['relevance']]}"
        )
    return "\n".join(out)


TOOLS = [web_search, fetch_website, find_similar_examples]
