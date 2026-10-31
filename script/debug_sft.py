"""Локальная отладка SFT-пайплайна (ноутбук 03) под trl 1.x на Apple MPS.

В trl 1.x убрали DataCollatorForCompletionOnlyLM: маскирование лосса теперь
делается через формат prompt/completion + completion_only_loss=True.
Скрипт проверяет этот путь целиком: данные -> маскирование -> обучение ->
сохранение адаптера -> генерация.

Запуск: python3 lm-hw/script/debug_sft.py
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"   # локальная отладка; на T4 будет 3B
SEED = 42

device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")
print("Устройство:", device)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float32,          # локально без квантования (bitsandbytes = CUDA only)
).to(device)

# --- данные: те же демо-примеры, что в ноутбуке 03 ---------------------------
DEMO_MESSAGES = [
    {"messages": [
        {"role": "system", "content": "Ты — помощник по бытовым вопросам дома и квартиры. Отвечай на русском языке, кратко и по делу. Если вопрос связан с опасностью (газ, электричество, протечки, нагрузки на конструкции), обязательно предупреди о рисках и укажи, когда нужно вызвать специалиста."},
        {"role": "user", "content": "Течёт батарея, что нужно делать?"},
        {"role": "assistant", "content": "Порядок действий при течи батареи:\n\n1. Подставьте ёмкость и накройте место течи тряпкой, чтобы вода не лилась на пол.\n2. Перекройте запорные краны на подводке к батарее (если они есть).\n3. Если кранов нет или течёт сам стояк — срочно звоните в управляющую компанию или аварийную службу (112).\n\nВНИМАНИЕ: не пытайтесь устранять течь на работающей системе под давлением — можно получить ожог. Ремонт радиатора доверьте сантехнику."}]},
    {"messages": [
        {"role": "system", "content": "Ты — помощник по бытовым вопросам дома и квартиры. Отвечай на русском языке, кратко и по делу. Если вопрос связан с опасностью (газ, электричество, протечки, нагрузки на конструкции), обязательно предупреди о рисках и укажи, когда нужно вызвать специалиста."},
        {"role": "user", "content": "В кухне пахнет газом, что делать?"},
        {"role": "assistant", "content": "При запахе газа действовать нужно немедленно:\n\n1. НЕ включайте и не выключайте электроприборы и свет — искра может вызвать взрыв.\n2. Перекройте кран подачи газа к плите.\n3. Откройте окна и покиньте помещение.\n4. С улицы позвоните в аварийную газовую службу: 104 или 112.\n\nОСТОРОЖНО: утечка газа — риск взрыва и отравления, самостоятельно искать место утечки нельзя."}]},
    {"messages": [
        {"role": "system", "content": "Ты — помощник по бытовым вопросам дома и квартиры. Отвечай на русском языке, кратко и по делу. Если вопрос связан с опасностью (газ, электричество, протечки, нагрузки на конструкции), обязательно предупреди о рисках и укажи, когда нужно вызвать специалиста."},
        {"role": "user", "content": "Дверь цепляется за напольное покрытие, как исправить?"},
        {"role": "assistant", "content": "Чаще всего дверь цепляет покрытие из-за просевших петель или вспучившегося ламината.\n\n1. Осмотрите зазор: если он равномерно маленький по низу — проблема в петлях, если в одном месте — в покрытии.\n2. Петли: ослабьте нижний винт петли, приподнимите полотно и затяните винт.\n3. Покрытие: проверьте стыки ламината, при вспучивании снимите плинтус и дайте полотнам место для расширения."}]},
    {"messages": [
        {"role": "system", "content": "Ты — помощник по бытовым вопросам дома и квартиры. Отвечай на русском языке, кратко и по делу. Если вопрос связан с опасностью (газ, электричество, протечки, нагрузки на конструкции), обязательно предупреди о рисках и укажи, когда нужно вызвать специалиста."},
        {"role": "user", "content": "Чем отмыть жир с кухонных шкафов без следов?"},
        {"role": "assistant", "content": "Для кухонных шкафов лучше всего работает тёплый раствор соды (2 ст. ложки на 0,5 л воды).\n\n1. Смочите мягкую тряпку и протрите фасады по направлению текстуры.\n2. Застарелый жир: нанесите кашицу из соды на 10–15 минут, затем смойте.\n3. Протрите насухо чистой тканью — иначе останутся разводы.\n\nНе используйте абразивы на глянцевых фасадах — останутся царапины."}]},
]

# --- формат prompt/completion (новый способ маскирования в trl 1.x) ----------
def to_prompt_completion(example):
    msgs = example["messages"]
    return {"prompt": msgs[:-1], "completion": [msgs[-1]]}

train_ds = Dataset.from_list(DEMO_MESSAGES).map(
    to_prompt_completion, remove_columns=["messages"]
)
print("Пример prompt:", json.dumps(train_ds[0]["prompt"], ensure_ascii=False)[:120])

# --- LoRA (без квантования) ---------------------------------------------------
lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
model.config.use_cache = False

# --- обучение ------------------------------------------------------------------
training_args = SFTConfig(
    output_dir=os.path.join(ROOT, "results", "sft_checkpoints_debug"),
    seed=SEED,
    max_length=1024,
    completion_only_loss=True,       # лосс только по completion (ответу ассистента)
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    num_train_epochs=1,
    max_steps=10,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=1,                 # warmup_ratio переименован в trl 1.x
    logging_steps=2,
    eval_strategy="steps",
    eval_steps=5,
    save_strategy="steps",
    save_steps=5,
    save_total_limit=2,
    report_to="none",
    use_cpu=(device == "cpu"),
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=train_ds.select([0, 1]),
    processing_class=tokenizer,
)

# --- ПРОВЕРКА МАСКИРОВАНИЯ (sanity check, обязателен) --------------------------
batch = next(iter(trainer.get_train_dataloader()))
labels = batch["labels"][0]
n_total = labels.numel()
n_masked = (labels == -100).sum().item()
n_supervised = (labels != -100).sum().item()
print(f"\nМаскирование: токенов {n_total}, замаскировано (промпт) {n_masked}, "
      f"с лоссом (ответ) {n_supervised}")
assert 0 < n_supervised < n_total, "Маскировка сломана!"
print("Маскирование работает корректно\n")

trainer.train()
print("\nОбучение завершено")

# --- сохранение адаптера --------------------------------------------------------
model.config.use_cache = True
adapter_dir = os.path.join(ROOT, "results", "sft_adapter_debug")
trainer.save_model(adapter_dir)
tokenizer.save_pretrained(adapter_dir)
print("Адаптер сохранён:", adapter_dir, "->", os.listdir(adapter_dir))

# --- генерация после обучения ----------------------------------------------------
model.eval()

@torch.no_grad()
def generate_answer(question: str) -> str:
    messages = [
        {"role": "system", "content": "Ты — помощник по бытовым вопросам дома и квартиры."},
        {"role": "user", "content": question},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    out = model.generate(**inputs, max_new_tokens=200, temperature=0.7,
                         top_p=0.9, do_sample=True, repetition_penalty=1.05)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

for q in ["Течёт батарея, что нужно делать?",
          "Дверь цепляется за напольное покрытие, как исправить?"]:
    print("=" * 80)
    print("ВОПРОС:", q)
    print("ОТВЕТ:", generate_answer(q))

print("\nOK: SFT-пайплайн (trl 1.x, prompt/completion) отработал целиком")
