"""Общие утилиты: метрики, работа с эмбеддингами, загрузка данных."""
import os
import gc
import hashlib
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sentence_transformers import SentenceTransformer

from config import (
    LABEL_MAP, EMBEDDING_MODEL, EMBEDDINGS_CACHE_DIR,
    RANDOM_STATE, VAL_SIZE
)


def load_data(train_path: str, eval_path: str):
    """Загружает и подготавливает данные."""
    train = pd.read_json(train_path, lines=True)
    eval_df = pd.read_json(eval_path, lines=True)
    
    train["label"] = train["relevance"].map(LABEL_MAP)
    eval_df["label"] = eval_df["relevance_new"].map(LABEL_MAP)
    
    return train, eval_df


def split_train_val(train: pd.DataFrame):
    """Разделяет train на train_part и val_part с группировкой по запросам."""
    gss = GroupShuffleSplit(n_splits=1, test_size=VAL_SIZE, random_state=RANDOM_STATE)
    tr_idx, val_idx = next(gss.split(train, groups=train["Text"]))
    return train.iloc[tr_idx], train.iloc[val_idx]


def evaluate(y_true, y_pred):
    """Считает метрики: 3-классовая accuracy, macro-F1 и бинарная accuracy."""
    return {
        "accuracy_3cls": accuracy_score(y_true, y_pred),
        "macro_f1_3cls": f1_score(y_true, y_pred, average="macro"),
        "accuracy_bin": accuracy_score(
            [1 if y == "RELEVANT" else 0 for y in y_true],
            [1 if y == "RELEVANT" else 0 for y in y_pred]
        ),
    }


def row_key(row):
    """Ключ для эмбеддинга строки."""
    return (f"query: {row['Text']} | рубрика: "
            f"{row['normalized_main_rubric_name_ru']} | {row['name']}")


def get_embeddings(df, cache_name="train_emb", model_name=EMBEDDING_MODEL):
    """Считает эмбеддинги с кэшированием."""
    EMBEDDINGS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    texts = [row_key(r) for _, r in df.iterrows()]
    fingerprint = hashlib.md5(
        (model_name + "||" + "\n".join(texts)).encode()
    ).hexdigest()
    
    npy_path = EMBEDDINGS_CACHE_DIR / f"{cache_name}.npy"
    fp_path = EMBEDDINGS_CACHE_DIR / f"{cache_name}.fingerprint"
    
    if npy_path.exists() and fp_path.exists():
        if fp_path.read_text() == fingerprint:
            print(f"Загружаю эмбеддинги из кэша: {npy_path}")
            return np.load(npy_path)
        else:
            print(f"Кэш устарел, пересчитываю")
    
    print("Считаю эмбеддинги (это может занять время)...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = 128
    if device == "cuda":
        model.half()
    
    emb = model.encode(
        texts, normalize_embeddings=True,
        batch_size=32, show_progress_bar=True
    ).astype(np.float32)
    
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    np.save(npy_path, emb)
    fp_path.write_text(fingerprint)
    print(f"Эмбеддинги сохранены: {npy_path}")
    
    return emb
