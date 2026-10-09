"""GSM8K prompts, training texts, exact-match metric."""
import re

from datasets import load_dataset

PROMPT_TEMPLATE = "Question: {question}\nAnswer:"
TRAIN_TEMPLATE = "Question: {question}\nAnswer: {answer}"
LABELLED_ANSWER = re.compile(r"####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)")
ANY_NUMBER = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")
CONTINUATION_MARKERS = ("\nQuestion:", "[Question]", "\nQ:", "\nProblem:")


def truncate_at_continuation(text: str) -> str:
    cut = len(text)
    for marker in CONTINUATION_MARKERS:
        index = text.find(marker)
        if index != -1:
            cut = min(cut, index)
    return text[:cut]


def load_gsm8k(split: str, limit=None):
    dataset = load_dataset("openai/gsm8k", "main", split=split)
    return dataset.select(range(limit)) if limit else dataset


def numbers_match(predicted, gold, tolerance: float = 1e-4) -> bool:
    if predicted is None or gold is None:
        return False
    try:
        x, y = float(predicted), float(gold)
    except (TypeError, ValueError):
        return predicted == gold
    return abs(x - y) <= tolerance * max(1.0, abs(y))


def extract_final_answer(text: str):
    text = truncate_at_continuation(text.strip())
    labelled = LABELLED_ANSWER.search(text)
    if labelled:
        return labelled.group(1).replace(",", "")
    numbers = ANY_NUMBER.findall(text)
    return numbers[-1].replace(",", "") if numbers else None


def build_training_texts(dataset, eos_token: str):
    return [TRAIN_TEMPLATE.format(question=q, answer=a) + eos_token
            for q, a in zip(dataset["question"], dataset["answer"])]


def build_prompts(dataset):
    return [PROMPT_TEMPLATE.format(question=q) for q in dataset["question"]]


def calibration_batches(dataset, dataset_subset, dataset_column, dataset_size, batch_size, tokenizer):
    """Replacement for ReplaceMe.utils.get_calib_dataloader: first dataset_size GSM8K train items, drop_last."""
    assert dataset == "openai/gsm8k", dataset
    data = load_dataset("openai/gsm8k", "main", split=dataset_subset)
    texts = [TRAIN_TEMPLATE.format(question=q, answer=a)
             for q, a in zip(data["question"], data["answer"])][:dataset_size]
    print(f"calibration: {len(texts)} texts, batch {batch_size}")
    return [texts[i:i + batch_size] for i in range(0, len(texts) - batch_size + 1, batch_size)]
