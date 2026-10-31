# LM HW: Дом и квартира

Узкоспециализированная LLM для бытовых вопросов по дому и квартире.
Высокая цена ошибки: газ, электрика, протечки.

**Базовая модель:** Qwen2.5-3B-Instruct (4-bit NF4) → SFT (QLoRA) → оценка до/после.

## Структура

```
├── README.md               # этот файл
├── notebooks/
│   ├── 01_baseline.ipynb   # прогон базовой модели на eval-сете
│   ├── 02_data_gen.ipynb   # сбор данных SFT
│   ├── 03_sft_train.ipynb  # QLoRA-обучение + wandb
│   └── 04_eval.ipynb       # оценка до/после (автопроверки + судья)
├── checks/
│   ├── autocheck.py        # автоматические проверки качества ответов
│   └── test_autocheck.py   # юнит-тесты срабатывания
├── judge/
│   └── judge_prompt.md     # зафиксированный промпт LLM-судьи
├── data/
│   ├── eval_questions.jsonl    # фиксированный eval-сет (100 вопросов, 48 опасных)
│   ├── sft_train.jsonl         # данные SFT (train)
│   └── sft_val.jsonl           # данные SFT (val)
└── results/                # ответы моделей, метрики, графики
```

## Запуск на T4 в Google Colab

Все ноутбуки рассчитаны на GPU T4 (16GB) в Google Colab.

1. Откройте ноутбук в Colab: `Runtime -> Change runtime type -> T4 GPU`.
2. Склонируйте репозиторий и установите зависимости (первая ячейка каждого ноутбука):

```python
!git clone https://github.com/<username>/lm-hw.git
%cd lm-hw
!pip install -q -U transformers datasets trl peft bitsandbytes accelerate wandb
```

3. Ноутбуки запускать по порядку номеров.

## Автопроверки

Проверки качества ответов (детектор предупреждений об опасности, язык, длина,
повторы) и их тесты:

```bash
python -m pytest checks/test_autocheck.py -v
```

## Данные

- `data/eval_questions.jsonl` — фиксированный набор из 100 вопросов для
  сравнения моделей «до/после» (расширен с 30 после первого бейзлайна, чтобы
  сузить доверительные интервалы метрик). Формат: `{"id": ..., "category": ..., "question": ..., "dangerous": true/false}`.
- `data/sft_train.jsonl`, `data/sft_val.jsonl` — данные для дообучения в chat-формате,
  совместимы с `datasets.load_dataset("json", data_files=...)`.

## Веса

LoRA-адаптер в формате safetensors: ссылка на Hugging Face Hub добавляется
после обучения (ноутбук 03).

## Метрики

- `safety_pass_rate` — доля опасных вопросов, на которые дан ответ с предупреждением;
- доля ответов на русском / без повторов / в рамках длины;
- баллы LLM-судьи: релевантность, полезность, безопасность (1-5).

Результаты сравнения «до/после»: `results/metrics.md` + таблицы в отчёте.
