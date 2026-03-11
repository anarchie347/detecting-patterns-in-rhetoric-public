import re
from typing import List, Tuple

_WS_RE = re.compile(r"\s+")

def _norm(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()       # Normalise whitespace, so that word counts are consistent and not affected by newlines/tabs/multiple spaces


def chunk_words(text: str, window: int = 250, stride: int = 125, min_words: int = 120) -> List[str]:
    """
    Split a document into overlapping word-window chunks.
    """
    text = _norm(text)
    words = text.split()
    if len(words) < min_words:
        return []

    chunks: List[str] = []
    for start in range(0, len(words), stride):      # Chunks overlap by stride, so that each chunk shares some words with the previous one, allowing the model to learn from more of the document context
        chunk = words[start:start + window]
        if len(chunk) < min_words:          
            break                       # If the remaining tail is too short, stops rather than returning a final tiny fragment
        chunks.append(" ".join(chunk))      
    return chunks


def chunk_dataset(
    docs: List[str],
    labels: List[int],
    window: int = 250,
    stride: int = 125,
    min_words: int = 120,       # Modify this default if you want to allow shorter chunks (e.g. for shorter documents)
) -> Tuple[List[str], List[int], List[str]]:
    """
    Convert document-level dataset -> chunk-level dataset.

    Inputs:
      docs:    list of full documents
      labels:  list of 0/1 labels per document

    Outputs:
      X_chunks: list[str] chunk texts
      y_chunks: list[int] labels per chunk
      groups:   list[str] doc_id per chunk (for leakage-safe CV later)
    """
    if len(docs) != len(labels):
        raise ValueError("docs and labels must be the same length")
    
    doc_ids = [f"doc_{i}" for i in range(len(docs))]        # doc_id is the grouping key: all chunks from the same original doc share it - will help avoid leakage during cross-validation later  

    X_chunks: List[str] = []
    y_chunks: List[int] = []
    groups: List[str] = []

    for doc, y, doc_id in zip(docs, labels, doc_ids):
        for c in chunk_words(doc, window=window, stride=stride, min_words=min_words):
            X_chunks.append(c)
            y_chunks.append(int(y))
            groups.append(doc_id)

    if not X_chunks:
        raise ValueError("No chunks produced. Lower min_words or check input texts.")

    return X_chunks, y_chunks, groups



## Example usage:   

"""
from chunking import chunk_dataset

# Provided by data cleaning step:
train_docs, train_labels = ...
test_docs, test_labels = ...

X_train, y_train, train_groups = chunk_dataset(train_docs, train_labels)
X_test,  y_test,  test_groups  = chunk_dataset(test_docs, test_labels)
"""