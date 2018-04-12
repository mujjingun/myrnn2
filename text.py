import numpy as np

def encode(text):
    encoding = "? abcdefghijklmnopqrstuvwxyz"
    return np.array([encoding.index(ch) for ch in text if ch in encoding] + [0], dtype=np.int32)
