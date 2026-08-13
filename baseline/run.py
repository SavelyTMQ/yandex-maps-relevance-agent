"""Запуск baseline."""
from tqdm import tqdm

from config import DATA_TRAIN_PATH, DATA_EVAL_PATH
from utils import load_data, evaluate
from baseline.model import QwenBaseline
from baseline.prompts import SYSTEM_PROMPT, build_user_prompt


def parse_label(text):
    # (тот же parse_label, что и в agent/run.py)
    ...


def main():
    train, eval_df = load_data(DATA_TRAIN_PATH, DATA_EVAL_PATH)
    
    model = QwenBaseline()
    
    preds = []
    for _, row in tqdm(eval_df.iterrows(), total=len(eval_df)):
        answer = model.predict(row, SYSTEM_PROMPT, build_user_prompt)
        preds.append(parse_label(answer))
    
    metrics = evaluate(eval_df["label"].tolist(), preds)
    print("\n=== Baseline результаты ===")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
