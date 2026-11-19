"""Build notebook 06 from pinned historical inputs and the tested CPU helper."""

import ast
import hashlib
import json
import subprocess
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMIT = 'f2768ea8029bc89bd1742e603440d6c5ac8756f4'
PILOT = 'results/sft_v3_pilot/qwen3_4b_20261118T113449101907'
DIAG = 'results/adapter_strength_v1/adapter_strength_v1_20261118T200443561241'
SOURCES = [
    'checks/__init__.py', 'checks/autocheck.py', 'checks/evaluation.py', 'checks/screening_v3.py',
    'data/sft_v3/selection.json', 'data/sft_v3/pilot_config.json',
    'data/sft_v3/adapter_strength_selection.json',
    'data/development_v3/questions.jsonl', 'data/development_v3/rubric.jsonl',
    'data/development_v3/sources.json',
    PILOT + '/answers.jsonl', PILOT + '/adapter/adapter_config.json',
    DIAG + '/answers.jsonl', DIAG + '/run_metadata.json',
    'judge/judge_prompt_v3.md', 'judge/judge_response_schema_v3.json',
    'judge/sft_v3_pilot/judge_requests.jsonl', 'judge/adapter_strength_v1/judge_requests.jsonl',
    'results/sft_v3_pilot/judges/qwen_scores.jsonl', 'results/sft_v3_pilot/judges/gemma_scores.jsonl',
    'results/adapter_strength_v1/judges/qwen_all_scales.jsonl',
    'results/adapter_strength_v1/judges/gemma_all_scales.jsonl',
]


def main():
    hashes = {}
    for relative in SOURCES:
        committed = subprocess.check_output(['git', 'show', f'{COMMIT}:{relative}'], cwd=ROOT)
        if (ROOT / relative).read_bytes() != committed:
            raise ValueError('Local input differs from pinned commit: ' + relative)
        hashes[relative] = hashlib.sha256(committed).hexdigest()
    helper = (ROOT / 'checks/adapter_expanded_v1.py').read_text()
    helper_sha = hashlib.sha256(helper.encode()).hexdigest()
    cells = []

    def md(source):
        cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': textwrap.dedent(source).strip() + '\n'})

    def code(source):
        source = textwrap.dedent(source).strip() + '\n'
        ast.parse(source)
        cells.append({'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': source})

    md('''
    # 06 — полный development для фиксированной LoRA 0,25

    **Google Colab → Среда выполнения → Сменить среду выполнения → T4 GPU.**
    Используйте новую сессию. Выполняйте восемь кодовых ячеек сверху вниз.

    Загрузите исходный архив **sft_v3_pilot_qwen3_4b_20261118T113449101907.zip**
    с весами. Архив с оценками судей не подходит. Все остальные данные
    автоматически загружаются из закреплённого коммита GitHub и проверяются по SHA-256.

    План: 32 прежних baseline + 8 прежних ответов LoRA 0,25 + **24 новых ответа**.
    Дополнительно выполняются шесть контрольных генераций. Обучения и запросов к судье нет.
    Условия прежние: 768 токенов, sampling, индивидуальный seed каждого вопроса.
    Это расширенный **development**, не независимый финальный тест.

    В конце скачиваются **два ZIP**: полный результат и отдельный пакет из 24 заданий судье.
    Разрешите браузеру скачивание нескольких файлов. Сохраните также выполненный IPYNB.
    ''')
    md('''
    ## 1. Установка версий прежнего успешного T4-прогона

    Если появится требование перезапустить Python: **Среда выполнения → Перезапустить сеанс**,
    затем снова выполняйте ячейки с первой. Менять версии для обхода проверки не нужно.
    ''')
    old_nb = json.loads((ROOT / 'notebooks/05_adapter_strength_diagnostic_v1.ipynb').read_text())
    install = [c for c in old_nb['cells'] if c['cell_type'] == 'code'][0]['source']
    install = ''.join(install) if isinstance(install, list) else install
    install = install.replace("'safetensors': '0.8.0'", "'safetensors': '0.8.0', 'numpy': '2.4.6'")
    install = install.replace('GPU-диагностика ещё не запускалась.', 'Генерация ещё не запускалась.')
    code(install)
    md('''
    ## 2. Источники, параметры и папка результата

    Можно оставить настройки как есть. Каждый ответ сразу сохраняется в `/content`.
    Для защиты от потери сессии поставьте `PERSIST_TO_DRIVE=True`: появится запрос
    подключить ваш Google Drive, результаты будут записываться туда после каждого ответа.

    После обрыва сессии результаты из `/content` исчезают. Если использовали Drive,
    укажите полный путь прежней папки запуска в `RESUME_DIR` и выполните ноутбук заново.
    Во время живой сессии для продолжения повторяют только ячейку 7.
    ''')
    init = '''
    import contextlib, hashlib, inspect, json, math, platform, random, shutil, sys, time, traceback, urllib.request, zipfile
    from datetime import datetime, timezone
    from pathlib import Path
    import numpy as np
    import torch
    import peft
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig, set_seed
    from peft import PeftModel, prepare_model_for_kbit_training
    from peft.tuners.lora.layer import LoraLayer

    PERSIST_TO_DRIVE = False
    RESUME_DIR = ''  # например /content/drive/MyDrive/lm_hw_expanded_runs/adapter_expanded_v1_...
    if 'model' in globals() or 'RUN_DIR' in globals():
        raise RuntimeError('Инициализация уже выполнена. Для продолжения повторите ячейку 7; для нового запуска создайте новую сессию.')
    if not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Выберите в Colab T4 GPU.')
    assert torch.__version__ == TORCH_PIN and torch.version.cuda == '13.0'
    assert peft.__version__ == PINS['peft'] and np.__version__ == PINS['numpy']
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    COMMIT = __COMMIT__
    SOURCE_HASHES = __HASHES__
    EMBEDDED_HELPER_SHA = __HELPER_SHA__
    SOURCE_ROOT = Path('/content/lm_hw_expanded_source_' + COMMIT[:7])
    for relative, expected in SOURCE_HASHES.items():
        destination = SOURCE_ROOT / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists() or hashlib.sha256(destination.read_bytes()).hexdigest() != expected:
            url = 'https://raw.githubusercontent.com/maria-igntv/lm-hw/' + COMMIT + '/' + relative
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read()
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError('SHA mismatch: ' + relative)
            destination.write_bytes(data)
    # A fresh Colab session must import exactly the pinned checks package.
    if 'checks' in sys.modules:
        raise RuntimeError('Модуль checks уже импортирован из другого места. Создайте новую сессию Python.')
    sys.path.insert(0, str(SOURCE_ROOT))
    selection = json.loads((SOURCE_ROOT / 'data/sft_v3/selection.json').read_text())
    settings = json.loads((SOURCE_ROOT / 'data/sft_v3/pilot_config.json').read_text())
    candidate_selection = json.loads((SOURCE_ROOT / 'data/sft_v3/adapter_strength_selection.json').read_text())
    questions = [json.loads(s) for s in (SOURCE_ROOT / 'data/development_v3/questions.jsonl').read_text().splitlines() if s.strip()]
    diagnostic_metadata = json.loads((SOURCE_ROOT / '__DIAG__/run_metadata.json').read_text())
    assert diagnostic_metadata['selection'] == selection and diagnostic_metadata['generation'] == settings['generation']
    assert candidate_selection['adapter_multiplier'] == 0.25
    assert candidate_selection['model_id'] == selection['model_id'] == 'Qwen/Qwen3-4B-Instruct-2507'
    assert candidate_selection['revision'] == selection['revision'] == 'cdbee75f17c01a7cc42f958dc650907174af0554'
    GENERATION = dict(settings['generation'])
    assert GENERATION == {'max_new_tokens':768, 'do_sample':True, 'temperature':0.7, 'top_p':0.8, 'top_k':20,
                          'min_p':0.0, 'repetition_penalty':1.0, 'num_beams':1, 'use_cache':True}
    CONTROL_IDS = ['dev3_01', 'dev3_15', 'dev3_25']
    EXPECTED_ADAPTER_SHA = candidate_selection['adapter_sha256']
    assert EXPECTED_ADAPTER_SHA == '461961f38e7552aa6d4c6990b5c1aaffea5801d24b1a4d4a4b86fda61feb98f8'
    question_index = {q['id']:q for q in questions}
    assert len(question_index) == len(questions) == 32
    SEEDS = {q['id']:settings['generation_seed'] + i for i,q in enumerate(questions, 1)}
    if PERSIST_TO_DRIVE or RESUME_DIR.startswith('/content/drive/'):
        from google.colab import drive
        drive.mount('/content/drive')
    OUTPUT_PARENT = Path('/content/drive/MyDrive/lm_hw_expanded_runs' if PERSIST_TO_DRIVE else '/content/lm_hw_expanded_runs')
    fixed_metadata = {'source_commit':COMMIT, 'source_hashes':SOURCE_HASHES, 'embedded_helper_sha256':EMBEDDED_HELPER_SHA,
        'selection':selection, 'candidate_selection':candidate_selection, 'generation':GENERATION,
        'adapter_multiplier':0.25, 'control_ids':CONTROL_IDS, 'seeds':SEEDS,
        'versions':{p:installed(p) for p in ['torch', *PINS]},
        'scope':'Expanded development of fixed scale 0.25; not a fresh final test; no training.'}
    if RESUME_DIR:
        RUN_DIR = Path(RESUME_DIR)
        metadata = json.loads((RUN_DIR / 'run_metadata.json').read_text())
        if any(metadata.get(k) != v for k,v in fixed_metadata.items()):
            raise ValueError('Возобновление остановлено: источники, версии или параметры прежнего запуска отличаются.')
        RUN_ID = metadata['run_id']
        assert RUN_DIR.name == RUN_ID
        metadata.setdefault('resumed_at_utc', []).append(datetime.now(timezone.utc).isoformat())
    else:
        RUN_ID = 'adapter_expanded_v1_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
        RUN_DIR = OUTPUT_PARENT / RUN_ID
        RUN_DIR.mkdir(parents=True, exist_ok=False)
        metadata = {**fixed_metadata, 'run_id':RUN_ID, 'status':'prepared_not_executed',
            'started_at_utc':datetime.now(timezone.utc).isoformat(),
            'hardware':{'gpu':torch.cuda.get_device_name(0), 'python':platform.python_version(), 'cuda':torch.version.cuda}}
    for relative in SOURCE_HASHES:
        target = RUN_DIR / 'source' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SOURCE_ROOT / relative, target)
    print('Источники проверены:', len(SOURCE_HASHES))
    print('Каталог результата:', RUN_DIR)
    print('План: 24 новых ответа + 6 контрольных; 40 ответов переиспользуются.')
    '''
    init = textwrap.dedent(init).replace('__COMMIT__', repr(COMMIT)).replace('__HASHES__', repr(hashes))
    init = init.replace('__HELPER_SHA__', repr(helper_sha)).replace('__DIAG__', DIAG)
    code(init)
    md('''
    ## 3. Проверенные функции сохранения и подготовки пакета судье

    Ячейку выполняют один раз. Она переносит прежние ответы без редактирования;
    уже сохранённые новые ответы при продолжении не генерируются повторно.
    ''')
    code(helper + '\nwrite_json(RUN_DIR / "run_metadata.json", metadata)\nprint("Функции сохранения готовы.")\n')
    md('''
    ## 4. Загрузите исходный ZIP SFT-пилота с весами

    По умолчанию появится выбор файла. Выберите
    **sft_v3_pilot_qwen3_4b_20261118T113449101907.zip** из Downloads на компьютере.
    Для уже загруженного файла укажите путь в `PILOT_ZIP_PATH`.
    Для пути в Drive сначала включите подключение Drive в ячейке 2.
    ''')
    code('''
    PILOT_ZIP_PATH = ''  # пусто: выбрать ZIP с компьютера; иначе путь к ZIP в /content или Drive
    ADAPTER_DIR = Path('/content/lm_hw_expanded_inputs') / RUN_ID / 'adapter'
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    if not (ADAPTER_DIR / 'adapter_model.safetensors').is_file():
        if PILOT_ZIP_PATH:
            user_zip = Path(PILOT_ZIP_PATH)
            if not user_zip.is_file(): raise FileNotFoundError(str(user_zip))
        else:
            from google.colab import files
            uploaded = files.upload()
            names = [n for n in uploaded if n.lower().endswith('.zip')]
            if len(names) != 1: raise ValueError('Выберите ровно один ZIP исходного SFT-пилота.')
            user_zip = Path('/content') / Path(names[0]).name
            if not user_zip.exists(): user_zip.write_bytes(uploaded[names[0]])
            del uploaded
        prefix = 'qwen3_4b_20261118T113449101907/adapter/'
        with zipfile.ZipFile(user_zip) as archive:
            for name in ['adapter_config.json', 'adapter_model.safetensors']:
                info = archive.getinfo(prefix + name)
                if info.file_size > 1_000_000_000: raise ValueError('Неожиданный размер адаптера')
                with archive.open(info) as source, (ADAPTER_DIR / name).open('wb') as dest:
                    shutil.copyfileobj(source, dest, 8 << 20)
        metadata['input_pilot_zip_sha256'] = sha256_file(user_zip)
    if sha256_file(ADAPTER_DIR / 'adapter_model.safetensors') != EXPECTED_ADAPTER_SHA:
        raise ValueError('SHA-256 адаптера отличается. Используйте исходный архив пилота, не tiny/smoke.')
    adapter_config = json.loads((ADAPTER_DIR / 'adapter_config.json').read_text())
    expected_config = json.loads((SOURCE_ROOT / '__PILOT__/adapter/adapter_config.json').read_text())
    assert adapter_config == expected_config
    assert adapter_config['r'] == 16 and adapter_config['lora_alpha'] == 32 and adapter_config['lora_dropout'] == 0
    assert not adapter_config['use_rslora'] and not adapter_config['use_dora']
    metadata['adapter_sha256'] = EXPECTED_ADAPTER_SHA
    write_json(RUN_DIR / 'adapter_config.json', adapter_config)
    write_json(RUN_DIR / 'inference_config.json', {'model_id':selection['model_id'], 'revision':selection['revision'],
        'adapter_sha256':EXPECTED_ADAPTER_SHA, 'adapter_multiplier':0.25,
        'scaling_method':'LoraLayer.set_scale(default, 0.25), absolute relative to alpha/r=2',
        'system_prompt':selection['system_prompt'], 'generation':GENERATION,
        'status':'development_candidate_not_final'})
    write_json(RUN_DIR / 'run_metadata.json', metadata)
    print('SHA-256 адаптера подтверждён:', EXPECTED_ADAPTER_SHA)
    '''.replace('__PILOT__', PILOT))
    md('''
    ## 5. Загрузка модели и одинаковых входов

    Модель, NF4/FP16, eager attention и подготовка параметров совпадают с прежним
    пилотом. Здесь нет оптимизатора и обновлений весов. Проверяются все 32 входа и seed.
    ''')
    code('''
    set_seed(settings['seed'])
    tokenizer = AutoTokenizer.from_pretrained(selection['model_id'], revision=selection['revision'], trust_remote_code=False)
    tokenizer.padding_side = 'right'
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    quant = dict(settings['quantization']); quant['bnb_4bit_compute_dtype'] = torch.float16
    base_model = AutoModelForCausalLM.from_pretrained(selection['model_id'], revision=selection['revision'],
        trust_remote_code=False, dtype=torch.float16, device_map={'':0},
        quantization_config=BitsAndBytesConfig(**quant), attn_implementation='eager').eval()
    assert base_model.__class__.__name__ == 'Qwen3ForCausalLM' and getattr(base_model, 'is_loaded_in_4bit', False)
    eos_ids = base_model.generation_config.eos_token_id
    if isinstance(eos_ids, int): eos_ids = [eos_ids]
    assert set(eos_ids) == {151645, 151643}
    base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant':False})
    base_model.config.use_cache = False
    model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR), adapter_name='default',
        is_trainable=False, autocast_adapter_dtype=True).eval()
    del base_model
    assert set(model.peft_config) == {'default'} and not any(p.requires_grad for p in model.parameters())
    assert all(p.dtype == torch.float32 for n,p in model.named_parameters() if 'lora_' in n)
    assert all(p.device.type == 'cuda' for p in model.parameters())
    generation_config = GenerationConfig(**GENERATION, eos_token_id=eos_ids, pad_token_id=tokenizer.pad_token_id)
    pilot_rows = read_jsonl(SOURCE_ROOT / '__PILOT__/answers.jsonl')
    diagnostic_rows = read_jsonl(SOURCE_ROOT / '__DIAG__/answers.jsonl')
    history = build_expanded_history(questions, settings, selection, pilot_rows, diagnostic_rows, RUN_ID)
    historical = {(r['prompt_id'],r['id']):r for r in history}
    canonical = {}
    for q in questions:
        qid = q['id']
        encoded = tokenizer.apply_chat_template([{'role':'system','content':selection['system_prompt']},
            {'role':'user','content':q['question']}], add_generation_prompt=True,
            tokenize=True, return_dict=True, return_tensors='pt')
        canonical[qid] = {k:v.cpu().clone() for k,v in encoded.items()}
        assert encoded['input_ids'][0].tolist() == historical['baseline',qid]['input_token_ids'], qid
        assert SEEDS[qid] == historical['baseline',qid]['seed'], qid
        assert encoded['input_ids'].shape[1] + 768 <= model.config.max_position_embeddings
    for row in history:
        assert row['input_token_ids'] == canonical[row['id']]['input_ids'][0].tolist()
        assert tokenizer.decode(row['generated_token_ids'], skip_special_tokens=True) == row['answer']
        assert tokenizer.decode(row['generated_token_ids'], skip_special_tokens=False) == row['raw_answer']
    write_json(RUN_DIR / 'input_audit.json', {qid:{'seed':SEEDS[qid],
        'input_token_ids':batch['input_ids'][0].tolist()} for qid,batch in canonical.items()})
    layers = {n:m for n,m in model.named_modules() if isinstance(m,LoraLayer) and 'default' in m.scaling}
    assert len(layers) == 252
    original_scales = {n:float(m.scaling['default']) for n,m in layers.items()}
    for name, layer in layers.items():
        assert not layer.merged and layer.r['default'] == 16 and layer.lora_alpha['default'] == 32
        assert original_scales[name] == 2.0
        assert tuple(inspect.signature(layer.set_scale).parameters) == ('adapter','scale')
    def apply_strength(multiplier):
        if multiplier not in [0.0, 0.25]: raise ValueError('В этом запуске разрешены только 0 и 0.25')
        for name, layer in layers.items():
            layer.set_scale('default', float(multiplier))
            assert math.isclose(float(layer.scaling['default']), original_scales[name] * multiplier, rel_tol=1e-12, abs_tol=1e-12)
    @contextlib.contextmanager
    def isolated_rng(seed):
        py, np_state = random.getstate(), np.random.get_state()
        cpu, cuda = torch.random.get_rng_state(), torch.cuda.get_rng_state_all()
        try:
            set_seed(seed)
            yield
        finally:
            random.setstate(py); np.random.set_state(np_state)
            torch.random.set_rng_state(cpu); torch.cuda.set_rng_state_all(cuda)
    @torch.inference_mode()
    def prompt_logits(qid):
        batch = {k:v.clone().to('cuda') for k,v in canonical[qid].items()}
        logits = model(**batch, use_cache=False, logits_to_keep=1).logits[0,-1].float().cpu()
        if not torch.isfinite(logits).all(): raise FloatingPointError('NaN/Inf logits')
        return logits
    def lora_fingerprint():
        return {n:hashlib.sha256(p.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                for n,p in model.named_parameters() if 'lora_' in n}
    @torch.inference_mode()
    def generate_current(qid, strength):
        apply_strength(strength)
        previous_cache = model.config.use_cache
        try:
            model.eval(); model.config.use_cache = True
            with isolated_rng(SEEDS[qid]):
                batch = {k:v.clone().to('cuda') for k,v in canonical[qid].items()}
                prefix = batch['input_ids'].shape[1]
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
                output = model.generate(**batch, generation_config=generation_config)
                torch.cuda.synchronize(); elapsed = time.perf_counter() - started
                ids = output[0,prefix:].tolist()
            return {**question_index[qid], 'adapter_multiplier':strength, 'seed':SEEDS[qid],
                'answer':tokenizer.decode(ids, skip_special_tokens=True),
                'raw_answer':tokenizer.decode(ids, skip_special_tokens=False),
                'generated_token_ids':ids, 'generated_tokens':len(ids), 'eos_token_ids':eos_ids,
                'max_new_tokens':768, 'finish_reason':'eos' if ids and ids[-1] in eos_ids else ('length' if len(ids)>=768 else 'other'),
                'input_token_ids':batch['input_ids'][0].tolist(), 'elapsed_seconds':elapsed,
                'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,
                'peak_reserved_gib':torch.cuda.max_memory_reserved()/1024**3}
        finally:
            model.config.use_cache = previous_cache
    apply_strength(0.25)
    _warmup = prompt_logits(CONTROL_IDS[0])
    lora_before = lora_fingerprint()
    write_json(RUN_DIR / 'lora_hashes_before.json', lora_before)
    write_json(RUN_DIR / 'scaling_audit.json', {'n_layers':len(layers), 'original_scales':original_scales,
        'target_multiplier':0.25, 'target_effective_scale':0.5,
        'set_scale_source_sha256':hashlib.sha256(inspect.getsource(next(iter(layers.values())).set_scale).encode()).hexdigest()})
    print('Модель готова. Все 32 входа и 40 прежних текстов проверены. Новые ответы ещё не сгенерированы.')
    '''.replace('__PILOT__', PILOT).replace('__DIAG__', DIAG))
    md('''
    ## 6. Проверка воспроизведения — шесть контрольных ответов

    Для трёх вопросов повторяются baseline и LoRA 0,25 с прежними seed.
    Требуется **точное совпадение выходных токенов**, а не только похожий текст.
    При расхождении запуск остановится: сохраните частичный ZIP ячейкой 8.
    Не отключайте проверку и не повторяйте генерацию в поиске удачного ответа.
    ''')
    code('''
    CONTROL_SESSION = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
    controls_path = RUN_DIR / ('controls_' + CONTROL_SESSION + '.jsonl')
    verification = {'status':'controls_running', 'control_session':CONTROL_SESSION,
                    'token_matches':{}, 'zero_vs_disabled':{}, 'scale_restoration':{}}
    write_json(RUN_DIR / 'verification.json', verification)
    try:
        apply_strength(0.25)
        quarter_logits = {q:prompt_logits(q) for q in CONTROL_IDS}
        apply_strength(0.0)
        zero_logits = {q:prompt_logits(q) for q in CONTROL_IDS}
        with model.disable_adapter():
            for qid in CONTROL_IDS:
                verification['zero_vs_disabled'][qid] = bool(torch.equal(zero_logits[qid], prompt_logits(qid)))
        apply_strength(0.25)
        for qid in CONTROL_IDS:
            verification['scale_restoration'][qid] = bool(torch.equal(quarter_logits[qid], prompt_logits(qid)))
        assert all(verification['zero_vs_disabled'].values()) and all(verification['scale_restoration'].values())
        controls = []
        for phase,strength in [('baseline',0.0), (EXPANDED_PHASE,0.25)]:
            for qid in CONTROL_IDS:
                row = generate_current(qid, strength)
                row.update(run_id=RUN_ID, control_session=CONTROL_SESSION, prompt_id=phase)
                controls.append(row); write_jsonl(controls_path, controls)
                matched = row['generated_token_ids'] == historical[phase,qid]['generated_token_ids']
                verification['token_matches'][phase + '/' + qid] = matched
                write_json(RUN_DIR / 'verification.json', verification)
                if not matched: raise RuntimeError('Исторические токены не воспроизвелись: ' + phase + '/' + qid + '. Сохраните ZIP ячейкой 8.')
                print('Контроль', len(controls), '/6:', phase, qid, 'MATCH', flush=True)
        verification['status'] = 'CONTROLS_PASS'
        metadata['status'] = 'controls_passed_not_complete'
        write_json(RUN_DIR / 'verification.json', verification)
        write_json(RUN_DIR / 'run_metadata.json', metadata)
        apply_strength(0.25)
        print('CONTROLS PASS. Можно генерировать недостающие 24 ответа.')
    except BaseException as exc:
        metadata['status'] = 'controls_failed'
        verification['status'] = 'CONTROLS_FAILED'
        write_json(RUN_DIR / 'verification.json', verification)
        write_json(RUN_DIR / 'run_metadata.json', metadata)
        write_jsonl(RUN_DIR / 'errors.jsonl', read_jsonl(RUN_DIR / 'errors.jsonl') + [{
            'stage':'controls', 'type':type(exc).__name__, 'message':str(exc), 'traceback':traceback.format_exc()}])
        raise
    ''')
    md('''
    ## 7. Основной прогон — 24 новых ответа

    Дождитесь **24/24**. Ответы сохраняются сразу после генерации.
    Если ячейка прервана, но сессия жива, повторите только её: готовые ответы будут пропущены.
    Для скачивания частичного результата можно выполнить ячейку 8 в любой момент после инициализации.
    ''')
    code('''
    def ensure_expanded_texts(rows):
        for row in rows:
            assert row['input_token_ids'] == canonical[row['id']]['input_ids'][0].tolist(), row['id']
            assert tokenizer.decode(row['generated_token_ids'], skip_special_tokens=True) == row['answer'], row['id']
            assert tokenizer.decode(row['generated_token_ids'], skip_special_tokens=False) == row['raw_answer'], row['id']
    def generate_new_answer(question, system, seed):
        assert system == selection['system_prompt'] and seed == SEEDS[question['id']]
        return generate_current(question['id'], 0.25)
    def on_new_saved(row, count):
        metadata.update(status='generating', new_answers_saved=count)
        write_json(RUN_DIR / 'run_metadata.json', metadata)
        print('Новых ответов:', count, '/24;', row['id'], row['finish_reason'], row['generated_tokens'], 'токенов', flush=True)
    def run_expanded():
        assert verification['status'] in ['CONTROLS_PASS', 'TECHNICAL_PASS_NOT_QUALITY_CLAIM']
        ensure_expanded_texts(read_jsonl(RUN_DIR / 'answers.jsonl'))
        try:
            metadata['status'] = 'generating'; write_json(RUN_DIR / 'run_metadata.json', metadata)
            rows = generate_expanded_remaining(RUN_DIR / 'answers.jsonl', questions, settings, selection,
                RUN_ID, history, generate_new_answer, on_new_saved)
            ensure_expanded_texts(rows)
            lora_after = lora_fingerprint()
            write_json(RUN_DIR / 'lora_hashes_after.json', lora_after)
            verification['lora_parameters_unchanged'] = lora_before == lora_after
            verification['adapter_file_unchanged'] = sha256_file(ADAPTER_DIR / 'adapter_model.safetensors') == EXPECTED_ADAPTER_SHA
            assert verification['lora_parameters_unchanged'] and verification['adapter_file_unchanged']
            verification['status'] = 'TECHNICAL_PASS_NOT_QUALITY_CLAIM'
            metadata.update(status='complete_awaiting_export', completed_at_utc=datetime.now(timezone.utc).isoformat(),
                            total_answers=64, new_answers_saved=24, reused_answers=40)
            write_json(RUN_DIR / 'verification.json', verification)
            write_json(RUN_DIR / 'run_metadata.json', metadata)
            print('ГОТОВО: 64 ответа = 32 baseline + 32 LoRA 0,25. Из них 24 новых. Качество ещё оценивает судья.')
        except BaseException as exc:
            metadata['status'] = 'generation_interrupted_or_failed'
            write_json(RUN_DIR / 'run_metadata.json', metadata)
            write_jsonl(RUN_DIR / 'errors.jsonl', read_jsonl(RUN_DIR / 'errors.jsonl') + [{
                'stage':'generation', 'type':type(exc).__name__, 'message':str(exc), 'traceback':traceback.format_exc()}])
            raise
    run_expanded()
    ''')
    md('''
    ## 8. Собрать и скачать результаты

    При успешном прогоне скачиваются:

    1. `adapter_expanded_v1_<timestamp>.zip` — все ответы, контроли, метаданные,
       источники, автоматические показатели и пакет судье. Этот ZIP верните для анализа.
    2. `adapter_expanded_v1_<timestamp>_for_judge.zip` — **только 24 новых задания**.
       Его можно сразу передать прежнему Qwen3.8-27B с конфигурацией внутри архива.

    Скачивание идёт через браузер на компьютер, обычно в **Downloads / Загрузки**.
    Если браузер блокирует второй файл, разрешите множественные загрузки или скачайте
    ZIP через панель **Файлы** в Colab по напечатанному пути.

    Сохраните ноутбук: **Файл → Скачать → Скачать файл IPYNB**.
    Полный результат не включает 125 MB весов: исходный ZIP пилота сохраните отдельно.
    Частичный ZIP при ошибке содержит статус и ошибки; пакет судье создаётся только после полного PASS.
    ''')
    code('''
    def package_expanded_run():
        rows = read_jsonl(RUN_DIR / 'answers.jsonl')
        check = json.loads((RUN_DIR / 'verification.json').read_text()) if (RUN_DIR / 'verification.json').exists() else {}
        complete = metadata['status'] in ['complete_awaiting_export','complete_awaiting_judge'] and check.get('status') == 'TECHNICAL_PASS_NOT_QUALITY_CLAIM'
        judge_zip = None
        if complete:
            rubrics = read_jsonl(SOURCE_ROOT / 'data/development_v3/rubric.jsonl')
            sources = json.loads((SOURCE_ROOT / 'data/development_v3/sources.json').read_text())['sources']
            count = export_expanded_artifacts(RUN_DIR, questions, rows, history, settings, selection, RUN_ID,
                rubrics, sources, (SOURCE_ROOT / 'judge/judge_prompt_v3.md').read_text(),
                json.loads((SOURCE_ROOT / 'judge/judge_response_schema_v3.json').read_text()))
            reused = export_expanded_reuse(RUN_DIR, SOURCE_ROOT, history, rubrics, sources)
            assert count == 24 and len(reused) == 80
            metadata.update(status='complete_awaiting_judge', judge_requests=24, reused_scores_per_judge=40)
            write_json(RUN_DIR / 'run_metadata.json', metadata)
            judge_zip = RUN_DIR.parent / (RUN_ID + '_for_judge.zip')
            shutil.copy2(RUN_DIR / 'judge/for_judge.zip', judge_zip)
        else:
            print('Сохраняется ЧАСТИЧНЫЙ результат. Пакет судье не создаётся. Статус:', metadata['status'])
        archive_path = RUN_DIR.parent / (RUN_ID + '.zip')
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(RUN_DIR.rglob('*')):
                if path.is_file(): archive.write(path, arcname=RUN_ID + '/' + str(path.relative_to(RUN_DIR)))
        print('Полный/частичный ZIP:', archive_path)
        if judge_zip: print('Пакет для Qwen3.8-27B:', judge_zip, '— 24 задания; повторно оценивать прежние 40 не нужно.')
        from google.colab import files
        files.download(str(archive_path))
        if judge_zip: files.download(str(judge_zip))
        return archive_path, judge_zip
    RESULT_ZIP, JUDGE_ZIP = package_expanded_run()
    ''')
    notebook = {'cells':cells, 'metadata':{'colab':{'name':'06_adapter_expanded_v1.ipynb', 'provenance':[]},
        'accelerator':'GPU', 'gpuClass':'standard',
        'kernelspec':{'display_name':'Python 3', 'language':'python', 'name':'python3'},
        'language_info':{'name':'python', 'version':'3.13'}}, 'nbformat':4, 'nbformat_minor':5}
    # nbformat >=4.5 requires stable unique cell IDs.
    for index, cell in enumerate(cells):
        cell['id'] = f'expanded-{index:02d}'
    target = ROOT / 'notebooks/06_adapter_expanded_v1.ipynb'
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n')
    preparation = {'status':'prepared_not_executed_on_t4', 'source_commit':COMMIT,
        'notebook_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
        'helper_sha256':helper_sha, 'source_hashes':hashes,
        'code_cells':sum(c['cell_type']=='code' for c in cells),
        'new_answers':24, 'reused_answers':40, 'sampling_control_answers':6,
        'new_judge_requests_per_judge':24, 'reused_scores_per_judge':40,
        'scope':'Fixed 0.25 expanded development; no new training or model/API calls during preparation.'}
    out = ROOT / 'results/adapter_expanded_v1'
    out.mkdir(exist_ok=True)
    (out / 'preparation.json').write_text(json.dumps(preparation, ensure_ascii=False, indent=2)+'\n')
    print(target)


if __name__ == '__main__':
    main()
