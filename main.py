import hazm
import re
import json
import pandas as pd
import glob
import numpy as np
from tqdm import tqdm
from collections import Counter
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
from IPython.display import display
from torch.utils.data import random_split
from sklearn.model_selection import train_test_split
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


# =====================================================
# 1. data
# =====================================================
def load_data(DATA_PATH):
    main_data = pd.read_csv(DATA_PATH, sep="\t")
    return main_data


def dataset_first_look(df: pd.DataFrame, sample_size: int = 2, text_columns=None):
    """Quick first-look analysis of a dataframe (for Persian poetry)."""

    line = "=" * 70
    print(f"\n{line}\nDATASET FIRST LOOK\n{line}")

    # ---------- 1) Shape ----------
    print("\n[1] SHAPE")
    print(f"    rows:    {len(df):,}")
    print(f"    columns: {len(df.columns)}")

    # ---------- 2) Columns ----------
    print("\n[2] COLUMNS")
    for i, col in enumerate(df.columns):
        print(f"    [{i}] {col!r}  dtype={df[col].dtype}")

    # ---------- 3) Null / empty ----------
    print("\n[3] NULL / EMPTY VALUES")
    for col in df.columns:
        n_empty = int(df[col].astype(str).str.strip().eq("").sum())
        pct = 100 * n_empty / len(df) if len(df) else 0
        status = "OK  " if n_empty == 0 else "WARN"
        print(f"    [{status}] {col}: {n_empty:,} empty ({pct:.2f}%)")

    # ---------- 4) Unique values ----------
    print("\n[4] UNIQUE VALUES PER COLUMN")
    for col in df.columns:
        n_unique = df[col].nunique(dropna=False)
        if n_unique <= 20:
            print(f"    - {col}: {n_unique} unique values")
            for val, count in Counter(df[col]).most_common():
                pct = 100 * count / len(df)
                print(f"        * {val!r}: {count:,} ({pct:.1f}%)")
        else:
            print(f"    - {col}: {n_unique:,} unique values")

    # ---------- 5) Poets distribution ----------
    if "poet" in df.columns:
        print("\n[5] POETS DISTRIBUTION")
        counts = df["poet"].value_counts()
        for poet, count in counts.head(30).items():
            pct = 100 * count / len(df)
            print(f"    - {poet}: {count:,} ({pct:.1f}%)")
        if len(counts) > 30:
            print(f"    ... and {len(counts) - 30} more poets")
        print(f"    total poets:            {len(counts)}")
        print(f"    poets with < 50 poems:  {(counts < 50).sum()}")
        print(f"    poets with >= 500:      {(counts >= 500).sum()}")
        print(f"    max / min ratio:        {counts.max() / counts.min():.1f}x")

    # ---------- 6) Text length distribution ----------
    print("\n[6] TEXT LENGTH DISTRIBUTION")

    if text_columns is None:
        text_columns = [c for c in df.columns if df[c].dtype == object]
    print(f"    (columns: {text_columns})")

    for col in text_columns:
        lengths = df[col].astype(str).str.len().to_numpy()
        if lengths.size == 0:
            continue
        print(f"    - {col}:")
        print(f"        * min:    {lengths.min()}")
        print(f"        * max:    {lengths.max()}")
        print(f"        * mean:   {lengths.mean():.1f}")
        print(f"        * median: {np.median(lengths):.1f}")
        print(f"        * std:    {lengths.std():.1f}")

        buckets = [0, 20, 50, 100, 200, 500, 1000, 2000, 5000, np.inf]
        labels = [
            "0-20",
            "20-50",
            "50-100",
            "100-200",
            "200-500",
            "500-1k",
            "1k-2k",
            "2k-5k",
            "5k+",
        ]
        hist = np.histogram(lengths, bins=buckets)[0]
        total = hist.sum() or 1
        for label, count in zip(labels, hist):
            pct = 100 * count / total
            bar = "#" * int(pct / 2)
            print(f"          {label:>7}: {count:>7,} ({pct:5.1f}%) {bar}")

    # ---------- 7) Class balance ----------
    print("\n[7] CLASS BALANCE")
    for col in df.columns:
        n_unique = df[col].nunique()
        if 1 < n_unique <= 20:
            counts = df[col].value_counts()
            imbalance = counts.max() / counts.min() if counts.min() > 0 else float("inf")
            print(f"    - {col}: imbalance = {imbalance:.2f}x")
            if imbalance > 5:
                print("        [WARN] heavily imbalanced")
            elif imbalance > 2:
                print("        [NOTE] slightly imbalanced")
            else:
                print("        [OK]   roughly balanced")

    # ---------- 8) Duplicates ----------
    print("\n[8] DUPLICATES")
    print(f"    - exact duplicate rows: {int(df.duplicated().sum()):,}")
    for col in text_columns:
        n = int(df[col].duplicated().sum())
        pct = 100 * n / len(df)
        print(f"    - duplicate values in {col!r}: {n:,} ({pct:.2f}%)")

    # ---------- 9) Samples ----------
    print("\n[9] SAMPLES")
    for i in range(min(sample_size, len(df))):
        row = df.iloc[i]
        print(f"\n    --- Sample {i + 1} ---")
        for col, val in row.items():
            val_str = str(val)
            if len(val_str) > 200:
                val_str = val_str[:200] + "..."
            print(f"        {col}: {val_str!r}")

    print(f"\n{line}\nAnalysis finished.\n{line}")


# =====================================================
# 2. preProcess
# =====================================================
_NORMALIZER = hazm.Normalizer(persian_numbers=False)


def num_normalizer(text: str) -> str:
    """Remove numbers as they are not important, and normalize the text."""

    normalized_text = _NORMALIZER.normalize(text)
    no_digits_text = re.sub(
        r"[0-9۰-۹٠-٩]+", "", normalized_text
    )  # latin, persian, arabic numbers
    return no_digits_text.strip()


def _normalize_for_dedup(text: str) -> str:
    text = text.replace("\u200c", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def remove_duplicates(df, subset=("poet", "poem"), keep="first", normalize=True):
    before = len(df)

    if normalize:
        key_cols = []
        for col in subset:
            tmp = f"__{col}_norm__"
            df[tmp] = df[col].astype(str).map(_normalize_for_dedup)
            key_cols.append(tmp)

        df = df.drop_duplicates(subset=key_cols, keep=keep)
        df = df.drop(columns=key_cols)
    else:
        df = df.drop_duplicates(subset=list(subset), keep=keep)

    df = df.reset_index(drop=True)
    after = len(df)

    # print(f"remove_duplicates: {before:,} -> {after:,} " f"(removed {before - after:,})")
    return df


# ==================== STEP 2: FORMAT (Qwen Native) ====================


def format_dataset_qwen(dataset):
    poems = []
    for poet, poem in zip(dataset["poet"], dataset["poem_clean"]):
        poem_formatted = re.sub(r"\s{2,}", "\n", poem).strip()

        prompt = f"<|im_start|>user\n{poet}<|im_end|>\n<|im_start|>assistant\n"
        response = f"{poem_formatted}<|im_end|>"

        poems.append({"prompt": prompt, "response": response})

    return pd.DataFrame(poems)


# =====================================================
# 3. model
# =====================================================
def load_model_and_tokenizer(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


# =====================================================
# 4.
# =====================================================
def train_val_spliter(dataset):
    train_df, val_df = train_test_split(
        dataset,
        test_size=0.1,
        random_state=42,
        shuffle=True,
    )

    return train_df, val_df


def HF_datset_convertor(dataset1, dataset2):
    hf_dataset1 = Dataset.from_pandas(dataset1, preserve_index=False)
    hf_dataset2 = Dataset.from_pandas(dataset2, preserve_index=False)

    return hf_dataset1, hf_dataset2


# =====================================================
# 5.pre tarin
# =====================================================
def process_datasets(train_dataset, val_dataset, tokenizer, max_length=512):
    def _tokenize_and_mask(batch):
        prompt_encodings = tokenizer(batch["prompt"], add_special_tokens=False)["input_ids"]
        response_encodings = tokenizer(batch["response"], add_special_tokens=False)[
            "input_ids"
        ]

        input_ids = []
        labels = []
        attention_mask = []

        for prompt_ids, response_ids in zip(prompt_encodings, response_encodings):
            combined_ids = prompt_ids + response_ids

            combined_labels = [-100] * len(prompt_ids) + response_ids

            if len(combined_ids) > max_length:
                combined_ids = combined_ids[:max_length]
                combined_labels = combined_labels[:max_length]

            input_ids.append(combined_ids)
            labels.append(combined_labels)
            attention_mask.append([1] * len(combined_ids))

        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

    train_dataset = train_dataset.map(
        _tokenize_and_mask, batched=True, remove_columns=train_dataset.column_names
    )

    val_dataset = val_dataset.map(
        _tokenize_and_mask, batched=True, remove_columns=val_dataset.column_names
    )

    print("✅ Data successfully tokenized and masked!")
    print(f"Train size: {len(train_dataset)} | Val size: {len(val_dataset)}")

    return train_dataset, val_dataset


import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def load_quantized_model(model_name):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )

    quantized_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map={"": "cuda:0"},
        trust_remote_code=True,
    )

    quantized_model = prepare_model_for_kbit_training(quantized_model)
    quantized_model.gradient_checkpointing_enable()

    return quantized_model


def apply_qlora(quantized_model):
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    qlora_model = get_peft_model(quantized_model, peft_config)
    qlora_model.config.use_cache = False

    trainable_params = sum(p.numel() for p in qlora_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in qlora_model.parameters())
    print("=" * 40)
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable %:          {100 * trainable_params / total_params:.4f}%")
    print("=" * 40)

    return qlora_model


if __name__ == "__main__":
    # ==================== CONFIG ==================
    DATA_PATH = "data/poems.tsv"
    MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

    # ==================== STEP 1: Load dataset ====================
    raw_data = load_data(DATA_PATH)
    # print(raw_data.head())

    # dataset_first_look(raw_data)

    # ==================== STEP 2: preProcess ====================
    remove_dup_data = remove_duplicates(raw_data, subset=("poet", "poem"), normalize=True)

    remove_dup_data["poem_clean"] = remove_dup_data["poem"].apply(num_normalizer)

    finall_dataset = format_dataset_qwen(remove_dup_data)

    print(finall_dataset.iloc[0]["prompt"])
    print(finall_dataset.iloc[0]["response"])

    # ==================== STEP 3: Load model & tokenizer ====================
    base_model, base_tokenizer = load_model_and_tokenizer(MODEL_NAME)

    # ==================== STEP 4: ====================
    train_data, valid_data = train_val_spliter(finall_dataset)
    # print(type(train_data))

    train_data, valid_data = HF_datset_convertor(train_data, valid_data)
    # print(type(train_data))
    # print(train_data[0])

    # ==================== STEP 5: ====================
    train_dataset, val_dataset = process_datasets(train_data, valid_data, base_tokenizer)

    # ==================== STEP 6: ====================
    quantized_model = load_quantized_model(MODEL_NAME)
    qlora_model = apply_qlora(quantized_model)
