
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset
from datasets import load_dataset

from config import (
    PROMPT_CLOSEDBOOK_EVAL,
    PROMPT_PASSAGE_TRAIN,
    PROMPT_QA_TRAIN,
)


# ═══════════════════════════════════════════════════════════════════
# SQuAD Passage Loader
# ═══════════════════════════════════════════════════════════════════

def load_squad_passages(
    split: str = "validation",
    seed: int = 42,
    max_passages: int = 0,
) -> List[Dict]:
    """
    Load SQuAD v2 and group questions by unique passage (context).

    Returns:
        List of dicts, each with:
            "context": str           — the passage text
            "questions": List[dict]  — each has "question" (str) and
                                       "answers" (List[str])

    Unanswerable questions (empty answer list) are dropped.
    Passages are shuffled deterministically by seed, then truncated
    to max_passages if > 0.
    """
    ds = load_dataset("rajpurkar/squad_v2", split=split)

    passage_map = defaultdict(list)
    for row in ds:
        # skip unanswerable questions — they have empty answer lists
        if not row["answers"]["text"]:
            continue
        passage_map[row["context"]].append({
            "question": row["question"],
            "answers": row["answers"]["text"],   # list of gold answers
        })

    passages = [
        {"context": ctx, "questions": qs}
        for ctx, qs in passage_map.items()
        if len(qs) > 0
    ]

    # deterministic shuffle
    rng = random.Random(seed)
    rng.shuffle(passages)

    if max_passages > 0:
        passages = passages[:max_passages]

    return passages


def get_all_questions(passages: List[Dict]) -> List[Dict]:
    """Flatten all questions from a list of passage dicts."""
    questions = []
    for p in passages:
        questions.extend(p["questions"])
    return questions


# ═══════════════════════════════════════════════════════════════════
# Training Datasets
# ═══════════════════════════════════════════════════════════════════

class PassageTrainDataset(Dataset):
    """
    Stage 2 training dataset — raw passage memorisation.

    Each sample is tokenised from:
        "Read the following passage and memorize the facts.\\nPassage: {context}"

    Labels = input_ids (standard causal-LM teacher forcing).
    """

    def __init__(self, passages: List[str], tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = [
            PROMPT_PASSAGE_TRAIN.format(context=p) for p in passages
        ]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),
        }


class QATrainDataset(Dataset):
    """
    Training dataset from Q&A pairs (synthetic or real).

    Each sample is tokenised from:
        "Answer the following question.\\nQuestion: {q}\\nAnswer: {a}"
    """

    def __init__(self, qa_pairs: List[Dict], tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = [
            PROMPT_QA_TRAIN.format(question=qa["question"], answer=qa["answer"])
            for qa in qa_pairs
        ]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),
        }


class CombinedTrainDataset(Dataset):
    """
    Stage 3 training dataset — passage text + synthetic Q&A pairs
    combined into a single dataset.
    """

    def __init__(
        self,
        passages: List[str],
        qa_pairs: List[Dict],
        tokenizer,
        max_length: int = 1024,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.texts = []
        for p in passages:
            self.texts.append(PROMPT_PASSAGE_TRAIN.format(context=p))
        for qa in qa_pairs:
            self.texts.append(
                PROMPT_QA_TRAIN.format(question=qa["question"], answer=qa["answer"])
            )

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),
        }
