# Выбранная конфигурация SFT v3

`selection.json` фиксирует `Qwen/Qwen3-4B-Instruct-2507`, ревизию,
бытовой промпт и основания выбора по двум независимым LLM-судьям.
[Разбор](../../results/model_screening_v3/judges/analysis.md).

`smoke_config.json` задаёт техническую проверку LoRA на T4;
`smoke_examples.jsonl` содержит четыре проверенных арифметических контроля
в формате messages system/user/assistant. Это входы проверки обучения.
Корпус из 80–120 проверенных бытовых ответов ещё предстоит подготовить.

Проверка выполняется в [02_sft_v3_smoke.ipynb](../../notebooks/02_sft_v3_smoke.ipynb).
[Инструкция запуска и сохранения](../../docs/sft_v3_smoke_guide.md).
Loss учитывает только ответ ассистента и EOS, маски реализованы в
`checks/sft_smoke_v3.py`. Полное обучение начинается заново с базовой модели.
