import torch

def build_digit_token_ids(tokenizer, max_conf=10):
    """Maps confidence values 0..10 to their token id (as they appear after 'is ')."""
    mapping = {}
    for v in range(max_conf + 1):
        ids = tokenizer.encode(f" {v}", add_special_tokens=False)
        mapping[v] = ids[0]
    return mapping

def find_confidence_token_index(tokenizer, full_ids, prompt_len, assistant_text, k):
    """Finds the absolute token index of the confidence digit k in the full sequence."""
    marker = f"is {k} out of 10"
    char_idx = assistant_text.find(marker)
    if char_idx == -1:
        return None
    digit_char_start = char_idx + len("is ")

    enc = tokenizer(assistant_text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc["offset_mapping"]

    for i, (start, end) in enumerate(offsets):
        if start <= digit_char_start < end:
            abs_idx = prompt_len + i
            if abs_idx < len(full_ids):
                return abs_idx
    return None