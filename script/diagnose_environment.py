"""Run the same offline FP16 base-model probe under separate library stacks.

Example:
  python3 script/diagnose_environment.py --label transformers522 --device mps

Run sequentially in each environment. No adapters or quantization are loaded.
Saved outputs are diagnostic records, not replacement baseline measurements.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
SYSTEM_PROMPTS = {
    "ru": (
        "Ты — помощник по бытовым вопросам дома и квартиры. Отвечай на русском "
        "языке, кратко и по делу. Если вопрос связан с опасностью (газ, электричество, "
        "протечки, нагрузки на конструкции), обязательно предупреди о рисках и укажи, "
        "когда нужно вызвать специалиста."
    ),
    "en": (
        "You are an assistant for household questions about homes and apartments. "
        "Answer in English, briefly and to the point. If a question involves danger "
        "(gas, electricity, water leaks, loads on structures), always warn about "
        "the risks and indicate when a professional should be called."
    ),
}


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--device", choices=["mps", "cpu", "cuda"], default="mps")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    args = parser.parse_args()
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable in this process. Check GPU access permissions.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")
    assert args.max_new_tokens > 0

    cases_path = ROOT / "data/diagnostics/environment_cases.jsonl"
    cases = [json.loads(line) for line in cases_path.read_text().splitlines() if line.strip()]
    checkpoint = (Path.home() / ".cache/huggingface/hub" /
                  "models--Qwen--Qwen2.5-3B-Instruct/snapshots" / REVISION)
    assert (checkpoint / "config.json").is_file(), "Pinned checkpoint must be cached locally."
    assert Path(args.label).name == args.label, "Label must be a directory name."
    output = ROOT / "results/diagnostics" / args.label
    output.mkdir(parents=True, exist_ok=True)
    dtype = torch.float32 if args.device == "cpu" else torch.float16
    generation = dict(max_new_tokens=args.max_new_tokens, do_sample=False,
                      repetition_penalty=1.05, use_cache=True)
    signature = dict(
        model=MODEL_NAME, revision=REVISION, device=args.device, dtype=str(dtype),
        attention="sdpa", quantization=None, adapter=None, generation=generation,
        prompts=SYSTEM_PROMPTS, cases_sha256=file_hash(cases_path),
        config_sha256=file_hash(checkpoint / "config.json"),
        versions={p: importlib.metadata.version(p) for p in
                  ("torch", "transformers", "tokenizers", "huggingface-hub", "safetensors")},
        python=platform.python_version(),
    )
    metadata_path = output / "metadata.json"
    if metadata_path.exists():
        assert json.loads(metadata_path.read_text())["signature"] == signature, "Run settings changed. Use a new label."
    else:
        metadata_path.write_text(json.dumps(dict(signature=signature,
            started_at=datetime.now(timezone.utc).isoformat()), ensure_ascii=False, indent=2) + "\n")

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(checkpoint), local_files_only=True, dtype=dtype, attn_implementation="sdpa",
    ).to(args.device).eval()
    model.config.use_cache = True
    shared_weights = model.get_input_embeddings().weight.data_ptr() == model.get_output_embeddings().weight.data_ptr()
    print(json.dumps(dict(label=args.label, **signature, embeddings_tied=shared_weights), ensure_ascii=False), flush=True)
    answers_path = output / "answers.jsonl"
    saved = [json.loads(line) for line in answers_path.read_text().splitlines()] if answers_path.exists() else []
    done = {r["id"] for r in saved}
    assert len(done) == len(saved) and done <= {c["id"] for c in cases}

    with torch.inference_mode(), answers_path.open("a", encoding="utf-8") as destination:
        for case in cases:
            if case["id"] in done:
                continue
            messages = [dict(role="system", content=SYSTEM_PROMPTS[case["language"]]),
                        dict(role="user", content=case["question"])]
            inputs = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt", return_dict=True,
            ).to(args.device)
            prompt_ids = inputs["input_ids"][0].tolist()
            set_seed(42)
            started = time.monotonic()
            generated = model.generate(**inputs, **generation)
            tokens = generated[0][len(prompt_ids):].tolist()
            record = dict(
                **case, answer=tokenizer.decode(tokens, skip_special_tokens=True),
                prompt_ids=prompt_ids, generated_ids=tokens,
                hit_token_limit=len(tokens) >= args.max_new_tokens,
                generation_seconds=round(time.monotonic() - started, 3),
            )
            destination.write(json.dumps(record, ensure_ascii=False) + "\n")
            destination.flush()
            os.fsync(destination.fileno())
            print(json.dumps({k: v for k, v in record.items() if k not in ("prompt_ids", "generated_ids")}, ensure_ascii=False), flush=True)
    complete = [json.loads(line) for line in answers_path.read_text().splitlines()]
    assert len(complete) == len(cases)
    print("COMPLETE", output, flush=True)


if __name__ == "__main__":
    run()
