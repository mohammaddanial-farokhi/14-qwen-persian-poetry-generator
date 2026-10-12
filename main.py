import hazm
import re
import json
import pandas as pd
import glob
import numpy as np
from tqdm import tqdm


# =====================================================
# 1. data
# =====================================================
def load_data(DATA_PATH):
    main_data = pd.read_csv(DATA_PATH, sep="\t")
    return main_data


# =====================================================
# 2. preProcess
# ====================================================
def num_normalizer(text: str) -> str:
    """Remove numbers as they are not important, and normalize the text."""

    _normalizer = hazm.Normalizer(persian_numbers=False)
    normalized_text = _normalizer.normalize(text)
    no_digits_text = re.sub(
        r"[0-9۰-۹٠-٩]+", "", normalized_text
    )  # latin, persian, arabic numbers
    return no_digits_text.strip()


if __name__ == "__main__":
    # ==================== CONFIG ==================
    DATA_PATH = "data/poems.tsv"

    # ==================== STEP 0: Load dataset ====================
    raw_data = load_data(DATA_PATH)
