# Опубликованные веса SFT v3

[GitHub Release](https://github.com/maria-igntv/lm-hw/releases/tag/artifacts-sft-v3-pilot) ·
[скачать sft_v3_pilot_adapter_only.zip](https://github.com/maria-igntv/lm-hw/releases/download/artifacts-sft-v3-pilot/sft_v3_pilot_adapter_only.zip).

Это исходный экспериментальный LoRA-checkpoint пилота, не финальная модель
с подтверждённым превосходством. Полный масштаб 1 ухудшил качество;
масштаб 0,25 проверяется отдельно. [Текущий результат](../results/adapter_expanded_v1/analysis.md).

В архиве есть:

- `qwen3_4b_20261118T113449101907/adapter/adapter_model.safetensors`;
- неизменённый `adapter_config.json` рядом с весами;
- `inference_config.json` с множителем 0,25, системным промптом и генерацией;
- README, manifest и SHA256SUMS.

SHA-256 весов:
`461961f38e7552aa6d4c6990b5c1aaffea5801d24b1a4d4a4b86fda61feb98f8`.
Размер safetensors — 132 187 888 байт. Хеш архива и перечень файлов:
[adapter_bundle_manifest.json](../results/weights/adapter_bundle_manifest.json).

Базовая модель скачивается отдельно: `Qwen/Qwen3-4B-Instruct-2507`, ревизия
`cdbee75f17c01a7cc42f958dc650907174af0554`. Карточка и условия модели —
[Hugging Face](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507).

Самый короткий способ воспроизвести опубликованное состояние — открыть
[ноутбук 06 в Colab](https://colab.research.google.com/github/maria-igntv/lm-hw/blob/main/notebooks/06_adapter_expanded_v1.ipynb),
выбрать T4 и загрузить ZIP из release при запросе файла. Внутренняя папка
сохранена в совместимом формате. Ноутбук проверяет SHA, конфигурацию и контроли.
Веса и исходная конфигурация не редактируются.

Простая загрузка `PeftModel.from_pretrained` даёт исходный масштаб 1.
Для проверяемой конфигурации нужно **после загрузки** установить 0,25
относительно исходного alpha/r=2 для всех 252 LoRA-слоёв:

```python
from peft.tuners.lora.layer import LoraLayer

layers = [layer for layer in model.modules()
          if isinstance(layer, LoraLayer) and "default" in layer.scaling]
assert len(layers) == 252
for layer in layers:
    layer.set_scale("default", 0.25)
    assert layer.scaling["default"] == 0.5
```

Для точного сравнения также нужны прежние подготовка к k-bit, FP16/NF4,
attention, промпт и seed — они сохранены в ноутбуке. Обучение заново описано
в ноутбуке 04; повторно обученный файл не гарантированно совпадёт побайтно
с опубликованным и не должен подменяться при проверке сохранённого checkpoint.
