import re
import string
import random
import torch
import numpy as np

def set_seed(seed: int = 42):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def extract_answer(text, target_format="baseline"):
    """
    Extracts the final answer from the generation.
    Returns (extracted_text, is_fallback).
    """
    if target_format == "baseline":
        marker = "Answer:"
        idx = text.rfind(marker)
        if idx != -1:
            return text[idx + len(marker):].strip(), False
    elif target_format == "finetuned":
        # Lenient match: case-insensitive, optional leading period/whitespace, allows "level"
        match = re.search(r'(?i)(.*?)(?:\.?\s*my confidence(?: level)? is)', text)
        if match and match.group(1).strip():
            return match.group(1).strip(), False
            
    # Fallback (used if marker/confidence phrase isn't found, or regex returns empty)
    sentences = re.split(r'(?<=[.!?]) +', text)
    if sentences:
        fallback = sentences[-1].strip()
    else:
        fallback = " ".join(text.split()[-15:])
    return fallback, True

def check_match(generation, aliases, use_old_method=False, target_format="baseline"):
    """Check if the normalized generation contains any of the normalized aliases."""
    if use_old_method:
        target_text = generation
        is_fallback = False
    else:
        target_text, is_fallback = extract_answer(generation, target_format=target_format)
        
    norm_gen = normalize_answer(target_text)
    for alias in aliases:
        if normalize_answer(alias) in norm_gen:
            return True, is_fallback
    return False, is_fallback

def parse_confidence(text):
    """
    Robust regex to parse confidence statement from model output.
    Looks for a number between 0 and 10 in patterns like "My confidence is 7 out of 10".
    Returns the integer confidence or None if malformed/missing.
    """
    # Look for patterns like X out of 10, or X/10, or confidence is X
    match = re.search(r'(?:confidence (?:is|level is) |(?:^|\s))(\d{1,2})(?:\s*(?:out of|/)\s*10)', text, re.IGNORECASE)
    if match:
        try:
            conf = int(match.group(1))
            if 0 <= conf <= 10:
                return conf
        except ValueError:
            pass
            
    # Fallback: just look for a number out of 10 explicitly
    match = re.search(r'(\d{1,2})\s*/\s*10', text)
    if match:
        try:
            conf = int(match.group(1))
            if 0 <= conf <= 10:
                return conf
        except ValueError:
            pass

    return None
