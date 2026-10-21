import os
import glob
import json
import re
import pandas as pd
import numpy as np
import hazm
import torch
from collections import Counter
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainerCallback,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM


# =====================================================
# 1. DATA LOADING & ANALYSIS
# =====================================================
def load_data(DATA_PATH):
    """Load the main dataset from a TSV file."""
    main_data = pd.read_csv(DATA_PATH, sep="\t")
    return main_data


def dataset_first_look(df: pd.DataFrame, sample_size: int = 2, text_columns=None):
    """Perform a quick initial analysis of the dataframe (for Persian poetry)."""
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
# 2. DATA PREPROCESSING
# =====================================================
_NORMALIZER = hazm.Normalizer(persian_numbers=False)


def num_normalizer(text: str) -> str:
    """Normalize Persian text and remove all digits (Latin, Persian, Arabic)."""
    normalized_text = _NORMALIZER.normalize(text)
    no_digits_text = re.sub(r"[0-9۰-۹٠-٩]+", "", normalized_text)
    return no_digits_text.strip()


def _normalize_for_dedup(text: str) -> str:
    """Helper to normalize text specifically for duplicate detection."""
    text = text.replace("\u200c", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def remove_duplicates(df, subset=("poet", "poem"), keep="first", normalize=True):
    """Remove duplicate rows based on specified columns with optional normalization."""
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
    return df


def format_for_sft(dataset):
    """Format the dataset into a single 'text' column using Qwen's chat template."""
    texts = []
    for poet, poem in zip(dataset["poet"], dataset["poem_clean"]):
        poem_formatted = re.sub(r"\s{2,}", "\n", poem).strip()
        prompt = f"<|im_start|>user\n{poet}<|im_end|>\n<|im_start|>assistant\n"
        response = f"{poem_formatted}<|im_end|>"
        texts.append(prompt + response)

    return pd.DataFrame({"text": texts})


# =====================================================
# 3. DATASET SPLITTING & CONVERSION
# =====================================================
def train_val_splitter(dataset):
    """Split the dataset into train and validation sets (90/10 ratio)."""
    train_df, val_df = train_test_split(
        dataset,
        test_size=0.1,
        random_state=42,
        shuffle=True,
    )
    return train_df, val_df


def hf_dataset_converter(dataset1, dataset2):
    """Convert pandas DataFrames into Hugging Face Dataset objects."""
    hf_dataset1 = Dataset.from_pandas(dataset1, preserve_index=False)
    hf_dataset2 = Dataset.from_pandas(dataset2, preserve_index=False)
    return hf_dataset1, hf_dataset2


# =====================================================
# 4. MODEL LOADING & QLORA SETUP
# =====================================================
def load_model_and_tokenizer(model_name):
    """Load a standard (non-quantized) model and tokenizer."""
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def load_quantized_model(model_name):
    """Load the base model with 4-bit quantization (QLoRA)."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    quantized_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="cuda:0",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    quantized_model = prepare_model_for_kbit_training(
        quantized_model,
        use_gradient_checkpointing=False,
    )
    return quantized_model


def apply_qlora(quantized_model):
    """Apply LoRA adapters to the quantized model for fine-tuning."""
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    qlora_model = get_peft_model(quantized_model, peft_config)
    qlora_model.config.use_cache = False

    return qlora_model


# =====================================================
# 5. TRAINING SETUP & EXECUTION
# =====================================================
def _print_training_checklist(trainer, training_args, expected_eval_steps):
    """Print a verification checklist before training starts."""
    print("\n" + "=" * 50)
    print("🔍 TRAINING SETUP VERIFICATION")
    print("=" * 50)
    print(f"1. Eval Strategy in Args : {training_args.eval_strategy}")
    print(f"2. Eval Steps in Args    : {training_args.eval_steps}")
    print(f"3. Save Steps in Args    : {training_args.save_steps}")

    try:
        callback_names = [type(cb).__name__ for cb in trainer.callback_handler.callbacks]
    except AttributeError:
        callback_names = ["Unknown (cannot access callbacks)"]

    print(f"4. Active Callbacks      : {callback_names}")

    if "CustomEvalCallback" in callback_names:
        print(
            f"✅ CustomEvalCallback is successfully attached (Forcing eval every {expected_eval_steps} steps)."
        )
    else:
        print(
            "❌ WARNING: CustomEvalCallback is MISSING! Evaluation will run at default steps."
        )
    print("=" * 50 + "\n")


def train_with_sft(
    model, tokenizer, train_dataset, val_dataset, final_output_dir, checkpoint_dir
):
    """Train the QLoRA model using SFTTrainer and save the final artifacts."""
    if os.path.exists(final_output_dir) and os.path.exists(
        os.path.join(final_output_dir, "adapter_config.json")
    ):
        print(f"Final model already exists at '{final_output_dir}'. Skipping training.")
        return

    class CustomEvalCallback(TrainerCallback):
        """Callback to force evaluation at custom intervals."""

        def __init__(self, eval_steps=1200):
            self.eval_steps = eval_steps
            print(
                f"✅ [Setup] CustomEvalCallback initialized with eval_steps={self.eval_steps}"
            )

        def on_step_end(self, args, state, control, **kwargs):
            """Trigger evaluation when reaching custom step intervals."""
            if state.global_step > 0 and state.global_step % self.eval_steps == 0:
                print(
                    f"\n🔄 [Callback Action] Step {state.global_step} reached. Forcing evaluation now...\n"
                )
                control.should_evaluate = True
            return control

    # Optimized training arguments
    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        optim="paged_adamw_8bit",
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        num_train_epochs=1,
        logging_steps=100,
        eval_strategy="no",
        eval_steps=1200,
        save_strategy="steps",
        save_steps=400,
        save_total_limit=2,
        save_safetensors=True,  # Prevent checkpoint corruption
        bf16=False,
        fp16=True,
        max_grad_norm=0.3,
        warmup_steps=100,
        report_to="none",
        dataloader_num_workers=0,
    )

    response_template_ids = tokenizer.encode(
        "<|im_start|>assistant\n", add_special_tokens=False
    )
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template_ids,
        tokenizer=tokenizer,
    )

    custom_eval_callback = CustomEvalCallback(eval_steps=1200)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=256,
        data_collator=collator,
        args=training_args,
        callbacks=[custom_eval_callback],
    )

    _print_training_checklist(trainer, training_args, expected_eval_steps=1200)

    # Safe training start with checkpoint resuming
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint-*"))
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
        print(f"Found previous checkpoints. Resuming from: {latest_checkpoint}")
        try:
            trainer.train(resume_from_checkpoint=latest_checkpoint)
        except Exception as e:
            print(f"\n⚠️ ERROR loading checkpoint: {e}")
            print("The checkpoint might be corrupted. Starting from scratch...\n")
            trainer.train()
    else:
        print("No checkpoints found. Starting Training from scratch...")
        trainer.train()

    print(f"Saving final model to {final_output_dir}...")
    trainer.model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    print("Training and saving complete!")


if __name__ == "__main__":
    # ==================== CONFIG ====================
    DATA_PATH = "data/poems.tsv"
    MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
    CHECK_POINTS_PATH = "data/model/checkpoints"
    FINAL_PEFT_MODEL = "data/model/final_model"

    # ==================== STEP 1: Load Dataset ====================
    raw_data = load_data(DATA_PATH)

    # ==================== STEP 2: Preprocess Data ====================
    remove_dup_data = remove_duplicates(raw_data, subset=("poet", "poem"), normalize=True)
    remove_dup_data["poem_clean"] = remove_dup_data["poem"].apply(num_normalizer)

    # Format into a single text column for SFTTrainer
    final_dataset = format_for_sft(remove_dup_data)

    # ==================== STEP 3: Load Tokenizer ====================
    base_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token

    # ==================== STEP 4: Split & Convert ====================
    train_data, valid_data = train_val_splitter(final_dataset)
    train_dataset, val_dataset = hf_dataset_converter(train_data, valid_data)

    # ==================== STEP 5: Load Quantized Model & Apply QLoRA ====================
    quantized_model = load_quantized_model(MODEL_NAME)
    qlora_model = apply_qlora(quantized_model)

    # ==================== STEP 6: Train with SFTTrainer ====================
    train_with_sft(
        model=qlora_model,
        tokenizer=base_tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        final_output_dir=FINAL_PEFT_MODEL,
        checkpoint_dir=CHECK_POINTS_PATH,
    )
