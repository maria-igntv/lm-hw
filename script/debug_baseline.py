"""Локальная отладка ноутбука 01_baseline.ipynb без CUDA (Apple MPS / CPU).

Повторяет ячейки ноутбука по порядку, но:
- устройство: cuda -> mps -> cpu (на Mac нет CUDA, bitsandbytes не работает);
- квантование 4-bit применяется только на CUDA;
- модель по умолчанию Qwen2.5-0.5B-Instruct (быстрая проверка кода);
  реальный прогон на T4 будет с Qwen2.5-3B.

Запуск: python3 script/debug_baseline.py [--model Qwen/Qwen2.5-3B-Instruct] [--n 2]
"""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
parser.add_argument("--n", type=int, default=2, help="сколько вопросов прогнать")
args = parser.parse_args()

# --- ячейка 1-2: устройство и модель ---------------------------------------
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Устройство: {device}")

tokenizer = AutoTokenizer.from_pretrained(args.model)
load_kwargs = {"device_map": device}
if device == "cuda":
    # только на CUDA: bitsandbytes требует CUDA
    from transformers import BitsAndBytesConfig

    load_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
else:
    load_kwargs["dtype"] = torch.float16 if device == "mps" else torch.float32

model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
model.eval()
print("Модель загружена:", args.model)

# --- ячейка 3: eval-сет ------------------------------------------------------
eval_set = [json.loads(l) for l in open(os.path.join(ROOT, "data", "eval_questions.jsonl"), encoding="utf-8")]
print(f"Вопросов: {len(eval_set)}, из них опасных: {sum(r['dangerous'] for r in eval_set)}")

# --- ячейка 4: генерация -----------------------------------------------------
SYSTEM_PROMPT = (
    "Ты — помощник по бытовым вопросам дома и квартиры. Отвечай на русском "
    "языке, кратко и по делу. Если вопрос связан с опасностью (газ, электричество, "
    "протечки, нагрузки на конструкции), обязательно предупреди о рисках и укажи, "
    "когда нужно вызвать специалиста."
)
GEN_KWARGS = dict(max_new_tokens=512, temperature=0.7, top_p=0.9,
                  do_sample=True, repetition_penalty=1.05)


@torch.no_grad()
def generate_answer(question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    out = model.generate(**inputs, **GEN_KWARGS)
    prompt_len = inputs["input_ids"].shape[1]
    return tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)


test_questions = ["Течёт батарея, что нужно делать?",
                  "Дверь цепляется за напольное покрытие, как исправить?"]
for q in test_questions:
    print("=" * 80)
    print("ВОПРОС:", q)
    print("ОТВЕТ:", generate_answer(q))

# --- ячейка 5: прогон по подмножеству eval-сета ------------------------------
results = []
for i, r in enumerate(eval_set[: args.n]):
    answer = generate_answer(r["question"])
    results.append({**r, "model": "baseline-debug", "answer": answer})
    print(f"[{i+1}/{args.n}] {r['question'][:50]}... -> {len(answer)} символов")

# --- ячейка 6: автопроверки ---------------------------------------------------
from checks.autocheck import check_dataset

summary = check_dataset(results)
print(json.dumps({k: v for k, v in summary.items() if k != "failures"},
                 indent=2, ensure_ascii=False))
print("\nПримеры нарушений:")
for f in summary["failures"][:10]:
    print(" -", f)
print("\nOK: конвейер 01_baseline отработал целиком")
