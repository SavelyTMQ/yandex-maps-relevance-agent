"""Общие настройки проекта."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Загружаем переменные из .env
load_dotenv()

# Корень проекта
PROJECT_ROOT = Path(__file__).parent

# Пути к данным
DATA_TRAIN_PATH = os.getenv("DATA_TRAIN_PATH", "./data/data_for_train.jsonl")
DATA_EVAL_PATH = os.getenv("DATA_EVAL_PATH", "./data/data_for_eval.jsonl")

# Кэш
CACHE_DIR = Path(os.getenv("CACHE_DIR", "./cache"))
CACHE_DIR.mkdir(exist_ok=True)

AGENT_CACHE_DIR = CACHE_DIR / "agent"
BASELINE_CACHE_DIR = CACHE_DIR / "baseline"
EMBEDDINGS_CACHE_DIR = CACHE_DIR / "embeddings"

# Маппинг меток
LABEL_MAP = {0.0: "IRRELEVANT", 0.1: "PARTIAL", 1.0: "RELEVANT"}
VALID_LABELS = {"RELEVANT", "PARTIAL", "IRRELEVANT"}

# Модели
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
BASELINE_MODEL = "Qwen/Qwen2.5-3B-Instruct"

# GigaChat
GIGACHAT_CREDENTIALS = os.getenv("GIGACHAT_CREDENTIALS")
GIGACHAT_MODEL = "GigaChat-2"

# Лимиты агента
MAX_TOOL_CALLS = 2
RECURSION_LIMIT = 15
MAX_RETRIES = 3

# Настройки данных
RANDOM_STATE = 42
VAL_SIZE = 0.15
