# utils/metrics.py
from jiwer import wer

def compute_wer(reference, hypothesis):
    return wer(reference, hypothesis)