import re
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel

# ==================== CONFIG ====================

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_PATH = "data/model/final_model"

MAX_NEW_TOKENS = 200
MIN_NEW_TOKENS = 60

TEMPERATURE = 0.75
TOP_P = 0.90
REPETITION_PENALTY = 1.08

# False: متن پیوسته را با تقسیم‌بندی تقریبی به مصراع تبدیل نکن.
# True: اگر مدل هیچ خط جدیدی تولید نکرد، یک تقسیم‌بندی تقریبی انجام بده.
APPROXIMATE_SPLIT_IF_NEEDED = True

# تعداد تقریبی واژه‌ها در هر مصراع در حالت تقسیم‌بندی تقریبی.
# این عدد مرز واقعی عروضی را تشخیص نمی‌دهد؛ فقط برای مرتب‌سازی ظاهری است.
MIN_WORDS_PER_HEMISTICH = 4
MAX_WORDS_PER_HEMISTICH = 11

HEMISTICH_SEPARATOR = "    "


# ==================== MODEL LOADING ====================


def load_model_for_inference():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model in 4-bit...")
    compute_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    # در صورت پشتیبانی مدل، cache سرعت تولید را بهتر می‌کند.
    if hasattr(model, "config"):
        model.config.use_cache = True

    print("Model loaded.")
    return model, tokenizer


# ==================== TEXT CLEANING ====================


def clean_generated_text(text):
    """Remove special chat markers while preserving line boundaries."""
    if not text:
        return ""

    text = text.replace("<|im_start|>assistant", "")
    text = text.replace("<|im_end|>", "")
    text = text.replace("<|endoftext|>", "")

    # حذف فاصله‌های نامرئی/غیرمعمولی که ممکن است از متن باقی مانده باشند
    text = text.replace("\u200b", "")
    text = text.replace("\ufeff", "")

    # حفظ newlineها؛ فاصله‌های داخل هر خط نرمال می‌شوند.
    lines = []
    for line in text.splitlines():
        line = re.sub(r"[ \t\u00a0]+", " ", line).strip()
        if line:
            lines.append(line)

    return "\n".join(lines).strip()


# ==================== APPROXIMATE FORMATTING ====================


def split_continuous_text_approximately(
    text,
    min_words=MIN_WORDS_PER_HEMISTICH,
    max_words=MAX_WORDS_PER_HEMISTICH,
):
    """
    Approximate visual formatting only.

    This does NOT detect Persian meter, true hemistich boundaries,
    meaning, or rhyme. It chooses candidate boundaries using punctuation
    and approximate word counts while preserving word order.
    """
    words = text.split()
    if not words:
        return []

    hemistiches = []
    current = []

    # واژه‌هایی که معمولاً بعدشان مکث طبیعی دیده می‌شود.
    pause_endings = ("،", "؛", "؟", "!", ":", "،")

    for index, word in enumerate(words):
        current.append(word)

        next_word_exists = index < len(words) - 1
        count = len(current)

        # در محدوده تقریبی، بعد از نشانه مکث برش بزن.
        punctuation_break = (
            word.endswith(pause_endings) and count >= min_words and next_word_exists
        )

        # اگر مصراع خیلی طولانی شد، برش اجباری ظاهری انجام بده.
        length_break = count >= max_words

        if punctuation_break or length_break:
            hemistiches.append(" ".join(current))
            current = []

    if current:
        hemistiches.append(" ".join(current))

    return hemistiches


def smart_poem_formatter(
    raw_text,
    approximate_split=APPROXIMATE_SPLIT_IF_NEEDED,
):
    """
    Preserve model-generated lines when available.
    If the model outputs one continuous line, optionally split it
    approximately and pair adjacent segments into displayed couplets.
    """
    text = clean_generated_text(raw_text)
    if not text:
        return ""

    lines = [line.strip() for line in text.splitlines() if line.strip()]

    # اگر مدل خط‌های جداگانه تولید کرده، همان مرزها را حفظ کن.
    if len(lines) >= 2:
        hemistiches = lines
    else:
        # متن پیوسته است؛ مرز واقعی مصراع‌ها معلوم نیست.
        if not approximate_split:
            return text

        hemistiches = split_continuous_text_approximately(text)

    # هر دو مصراع در یک خط با چهار فاصله از هم.
    formatted_lines = []
    for i in range(0, len(hemistiches), 2):
        first = hemistiches[i].strip()

        if i + 1 < len(hemistiches):
            second = hemistiches[i + 1].strip()
            formatted_lines.append(f"{first}{HEMISTICH_SEPARATOR}{second}")
        else:
            # مصراع باقی‌مانده را حذف نکن.
            formatted_lines.append(first)

    return "\n".join(formatted_lines)


# ==================== GENERATION ====================


def generate_poem(
    model,
    tokenizer,
    poet_name,
    max_new_tokens=MAX_NEW_TOKENS,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    show_raw=True,
):
    poet_name = poet_name.strip()
    if not poet_name:
        raise ValueError("نام شاعر نمی‌تواند خالی باشد.")

    # این prompt با قالبی که در آموزش استفاده کرده‌ای مطابقت دارد.
    prompt = f"<|im_start|>user\n{poet_name}<|im_end|>\n" f"<|im_start|>assistant\n"

    inputs = tokenizer(prompt, return_tensors="pt")

    # با device_map='auto' ورودی‌ها باید به دستگاه ورودی مدل منتقل شوند.
    input_device = model.get_input_embeddings().weight.device
    inputs = {key: value.to(input_device) for key, value in inputs.items()}

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id == tokenizer.unk_token_id:
        im_end_id = tokenizer.eos_token_id

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": temperature,
        "top_p": top_p,
        "repetition_penalty": REPETITION_PENALTY,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": im_end_id,
        "use_cache": True,
    }

    with torch.inference_mode():
        outputs = model.generate(**inputs, **generation_kwargs)

    generated_tokens = outputs[0, inputs["input_ids"].shape[1] :]
    raw_poem = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    if show_raw:
        print("\n========== RAW MODEL OUTPUT ==========\n")
        print(repr(raw_poem))
        print("\n======================================\n")

    formatted_poem = smart_poem_formatter(raw_poem)

    return formatted_poem


# ==================== MAIN ====================

if __name__ == "__main__":
    model, tokenizer = load_model_for_inference()

    poet = input("نام شاعر را وارد کن (مثلاً حافظ): ").strip()

    poem = generate_poem(
        model=model,
        tokenizer=tokenizer,
        poet_name=poet,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        show_raw=True,
    )

    print(f"\n{'=' * 20} شعر تولیدشده: {poet} {'=' * 20}\n")
    print(poem)
    print("\n" + "=" * 60)
