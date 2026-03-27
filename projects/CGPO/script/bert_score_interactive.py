#!/usr/bin/env python3
import os
import sys

import torch
from bert_score import BERTScorer
from transformers import AutoModel, AutoTokenizer


def build_encoder():
    """Load a vanilla BERT encoder once for reuse."""
    model_name = os.getenv("BERT_MODEL", "bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def build_scorer() -> BERTScorer:
    """Build a reusable BERTScorer to avoid reloading the model each query."""
    lang = os.getenv("BERT_SCORE_LANG", "zh")
    model_type = os.getenv("BERT_SCORE_MODEL", "")
    if model_type:
        return BERTScorer(model_type=model_type, lang=lang)
    return BERTScorer(lang=lang)


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    masked = last_hidden_state * mask
    summed = masked.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def encode(text: str, tokenizer, model) -> torch.Tensor:
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return mean_pool(outputs.last_hidden_state, inputs["attention_mask"])


def main() -> int:
    print("BERT-Score + BERT embedding similarity mode. Press Ctrl+C or Ctrl+D to exit.")
    tokenizer, model = build_encoder()
    scorer = build_scorer()

    while True:
        try:
            text_a = input("Text1> ")
        except EOFError:
            print()
            break
        if not text_a:
            print("Text1 is empty, please input again.")
            continue

        try:
            text_b = input("Text2> ")
        except EOFError:
            print()
            break
        if not text_b:
            print("Text2 is empty, please input again.")
            continue

        emb_a = encode(text_a, tokenizer, model)
        emb_b = encode(text_b, tokenizer, model)
        cosine = torch.nn.functional.cosine_similarity(emb_a, emb_b).item()
        precision, recall, f1 = scorer.score([text_a], [text_b])
        p_val = precision[0].item()
        r_val = recall[0].item()
        f_val = f1[0].item()
        print(f"Cosine similarity -> {cosine:.6f}")
        print(f"BERT-Score -> P: {p_val:.6f} R: {r_val:.6f} F1: {f_val:.6f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
