"""Промпт для baseline (без агентности)."""

SYSTEM_PROMPT = """Ты — опытный асессор Яндекс.Карт..."""  # ваш промпт целиком

def build_user_prompt(row) -> str:
    return f"""Запрос: {row['Text']}

Организация:
- Название: {row['name']}
- Основная рубрика: {row['normalized_main_rubric_name_ru']}
- Адрес: {row['address']}
- Услуги и цены: {str(row.get('prices_summarized', ''))[:1500] or '—'}
- Сводка отзывов: {str(row.get('reviews_summarized', ''))[:2500] or '—'}"""
