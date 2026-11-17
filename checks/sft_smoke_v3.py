"""Construct assistant-only labels for the fixed Qwen training smoke."""


def build_masked_example(tokenizer, record, max_seq_length):
    messages = record['messages']
    if [message['role'] for message in messages] != ['system', 'user', 'assistant']:
        raise ValueError('Expected a single system/user/assistant example')
    prefix = tokenizer.apply_chat_template(
        messages[:2], tokenize=True, return_dict=False, add_generation_prompt=True,
    )
    full = tokenizer.apply_chat_template(
        messages, tokenize=True, return_dict=False, add_generation_prompt=False,
    )
    if full[:len(prefix)] != prefix:
        raise ValueError('Assistant boundary differs between prompt and full chat tokenization')
    end_positions = [index for index, token_id in enumerate(full) if token_id == tokenizer.eos_token_id]
    if not end_positions or end_positions[-1] < len(prefix):
        raise ValueError('Assistant message-end EOS is missing')
    full = full[:end_positions[-1] + 1]
    if len(full) > max_seq_length:
        raise ValueError('Control example exceeds max_seq_length')
    supervised = full[len(prefix):]
    if len(supervised) < 2:
        raise ValueError('Assistant target must include content and EOS')
    padding = max_seq_length - len(full)
    batch = {
        'input_ids': full + [tokenizer.pad_token_id] * padding,
        'attention_mask': [1] * len(full) + [0] * padding,
        'labels': [-100] * len(prefix) + supervised + [-100] * padding,
    }
    audit = {
        'sample_id': record['sample_id'], 'prompt_tokens': len(prefix),
        'unpadded_tokens': len(full), 'padded_tokens': max_seq_length,
        'supervised_tokens': len(supervised), 'last_supervised_token_id': supervised[-1],
        'supervised_text': tokenizer.decode(supervised, skip_special_tokens=False),
    }
    return batch, audit
