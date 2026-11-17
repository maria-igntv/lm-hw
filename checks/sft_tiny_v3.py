"""Shared evaluation and optimizer loop for the eight-example LoRA check."""

import random
import time


def select_training_records(records, sample_ids, system_prompt):
    index = {record['sample_id']: record for record in records}
    if len(index) != len(records) or len(set(sample_ids)) != 8:
        raise ValueError('Expected unique corpus IDs and eight distinct selected IDs')
    selected = [index[sample_id] for sample_id in sample_ids]
    if any(record['split'] != 'train' for record in selected):
        raise ValueError('Tiny training may only use train records')
    if len({record['category'] for record in selected}) != 8 or sum(record['dangerous'] for record in selected) != 4:
        raise ValueError('Expected eight categories and a four/four risk balance')
    if any(record['messages'][0]['content'] != system_prompt for record in selected):
        raise ValueError('Selected examples have a different system prompt')
    return selected


def supervised_token_count(batch):
    """Causal LM loss compares logits[:-1] with labels[1:]."""
    return sum(label != -100 for label in batch['labels'][1:])


def batch_to_device(batch, device):
    import torch
    return {key: torch.tensor([value], dtype=torch.long, device=device) for key, value in batch.items()}


def evaluate_model(model, batches, sample_ids, device, compute_dtype):
    import torch
    device = torch.device(device)
    was_training = model.training
    rows = []
    model.eval()
    try:
        with torch.inference_mode():
            for batch, sample_id in zip(batches, sample_ids, strict=True):
                tensors = batch_to_device(batch, device)
                with torch.autocast(device_type=device.type, dtype=compute_dtype, enabled=device.type == 'cuda'):
                    output = model(**tensors, use_cache=False)
                if not torch.isfinite(output.loss):
                    raise FloatingPointError('Evaluation loss contains NaN or Inf')
                labels = tensors['labels'][:, 1:]
                mask = labels != -100
                predictions = output.logits[:, :-1].argmax(dim=-1)
                count = int(mask.sum().item())
                if count == 0:
                    raise ValueError('Evaluation example has no supervised tokens')
                last_position = mask[0].nonzero(as_tuple=False)[-1, 0]
                rows.append({
                    'sample_id': sample_id, 'loss': float(output.loss.float().cpu()),
                    'supervised_tokens': count,
                    'correct_tokens': int(((predictions == labels) & mask).sum().item()),
                    'eos_correct': bool((predictions[0, last_position] == labels[0, last_position]).item()),
                })
                del tensors, output, labels, mask, predictions
    finally:
        model.train(was_training)
    total = sum(row['supervised_tokens'] for row in rows)
    if not total:
        raise ValueError('Evaluation set is empty')
    return {
        'n': len(rows), 'supervised_tokens': total,
        'token_weighted_loss': sum(row['loss'] * row['supervised_tokens'] for row in rows) / total,
        'token_accuracy': sum(row['correct_tokens'] for row in rows) / total,
        'eos_accuracy': sum(row['eos_correct'] for row in rows) / len(rows), 'per_example': rows,
    }


def train_tiny_model(model, batches, sample_ids, settings, device, compute_dtype, on_step, on_evaluation):
    import torch
    device = torch.device(device)
    accumulation = settings['gradient_accumulation_steps']
    if not batches or len(batches) != len(sample_ids) or len(batches) % accumulation:
        raise ValueError('Tiny set must contain complete accumulation groups')
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters or any(parameter.dtype != torch.float32 for parameter in parameters):
        raise ValueError('Expected trainable float32 adapter parameters')
    if any(parameter.requires_grad for name, parameter in model.named_parameters() if 'lora_' not in name):
        raise ValueError('Base weights must be frozen')
    optimizer = torch.optim.AdamW(parameters, lr=settings['learning_rate'],
                                  weight_decay=settings['weight_decay'], foreach=False)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda', init_scale=128.0)
    rng = random.Random(settings['seed'])
    order = []
    cursor = 0
    history = []
    model.train()
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    try:
        for step in range(1, settings['optimizer_steps'] + 1):
            if cursor == len(order):
                order = list(range(len(batches)))
                rng.shuffle(order)
                cursor = 0
            chosen = order[cursor:cursor + accumulation]
            cursor += accumulation
            total_tokens = sum(supervised_token_count(batches[index]) for index in chosen)
            if total_tokens == 0:
                raise ValueError('Accumulation group has no supervised tokens')
            optimizer.zero_grad(set_to_none=True)
            microbatches = []
            for index in chosen:
                count = supervised_token_count(batches[index])
                tensors = batch_to_device(batches[index], device)
                with torch.autocast(device_type=device.type, dtype=compute_dtype, enabled=device.type == 'cuda'):
                    output = model(**tensors, use_cache=False)
                    loss = output.loss
                if not torch.isfinite(loss):
                    raise FloatingPointError('Training loss contains NaN or Inf')
                scaler.scale(loss * count / total_tokens).backward()
                microbatches.append({'sample_id': sample_ids[index], 'loss': float(loss.detach().cpu()),
                                     'supervised_tokens': count})
                del tensors, output, loss
            scaler.unscale_(optimizer)
            gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
            if not gradients or not all(torch.isfinite(gradient).all().item() for gradient in gradients):
                raise FloatingPointError('Gradients are absent or contain NaN / Inf')
            norm = torch.nn.utils.clip_grad_norm_(parameters, settings['max_grad_norm'], error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            row = {
                'step': step, 'microbatches': microbatches, 'supervised_tokens': total_tokens,
                'token_weighted_loss': sum(item['loss'] * item['supervised_tokens'] for item in microbatches) / total_tokens,
                'gradient_norm_before_clip': float(norm.detach().cpu()), 'scaler_scale': scaler.get_scale(),
                'elapsed_seconds': time.perf_counter() - started,
            }
            history.append(row)
            on_step(row)
            if step in settings['evaluation_steps']:
                on_evaluation(step)
    finally:
        optimizer.zero_grad(set_to_none=True)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    summary = {
        'optimizer_steps': len(history), 'microbatches': len(history) * accumulation,
        'equivalent_passes_over_tiny_set': len(history) * accumulation / len(batches),
        'training_wall_seconds_including_evaluations': time.perf_counter() - started,
        'nonzero_B_matrices': sum(torch.count_nonzero(parameter).item() > 0
                                 for name, parameter in model.named_parameters() if 'lora_B' in name),
    }
    if device.type == 'cuda':
        summary['peak_allocated_gib'] = torch.cuda.max_memory_allocated(device) / 1024**3
        summary['peak_reserved_gib'] = torch.cuda.max_memory_reserved(device) / 1024**3
    return summary
