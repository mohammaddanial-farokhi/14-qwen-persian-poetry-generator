# 🌹 Persian Poetry Generator (Qwen2.5-3B)

![Python](https://img.shields.io/badge/Python-3.12.13-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1%2Bcu121-red.svg)
![Transformers](https://img.shields.io/badge/Transformers-4.45.2-yellow.svg)
![PEFT](https://img.shields.io/badge/PEFT-QLoRA-orange.svg)
![Gradio](https://img.shields.io/badge/UI-Gradio-ff69b4.svg)
![Hardware](https://img.shields.io/badge/Hardware-RTX%204060%20(8GB)-green.svg)
![Task](https://img.shields.io/badge/Task-Poetry%20Generation-success.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

---

## 🧠 About The Project

A complete end-to-end pipeline to fine-tune the **Qwen2.5-3B-Instruct** language model using **QLoRA** for generating Persian poetry in the style of legendary Iranian poets (such as Hafez, Rumi, Saadi, and Saib Tabrizi). 

This project is specifically optimized for consumer hardware and completes the entire training process on an **8GB VRAM GPU (RTX 4060 Laptop)** . Additionally, it features a fully functional web interface built with **Gradio** for easy user interaction.

**Result:** The training loss dropped from **2.97** to **2.55**, with a stable validation loss (Eval Loss: 2.57), demonstrating successful convergence without suffering from overfitting.

---

## 🎯 Key Features

1. **Smart Data Cleaning:** Utilizes the `hazm` library for Persian text normalization, removing digits, and standardizing half-spaces (ZWNJ).
2. **Deduplication:** Prevents the model from memorizing exact poems by removing duplicate entries.
3. **ChatML Formatting:** Converts raw poems into Qwen's chat template format (`<|im_start|>user...`) for optimal instruction-following.
4. **Loss Masking:** Computes loss *only* on the poem text, not the poet's name, ensuring the model learns to generate poetry rather than echoing prompts.
5. **QLoRA Fine-Tuning:** Uses 4-bit quantization and trains only ~1.73% of the model's parameters to fit into limited VRAM.
6. **Checkpoint Resume:** Robust training script that automatically detects and resumes from the latest checkpoint safely.
7. **Smart Post-processing:** A custom algorithm to format the model's continuous text output into standard Persian couplets (hemistiches separated by 4 spaces).

---

## 📂 Project Structure

```text
.
├── data/
│   ├── model/
│   │   ├── checkpoints/       # Training checkpoints (saved every 400 steps)
│   │   └── final_model/       # Final LoRA adapter and tokenizer
│   └── poems.tsv              # Raw dataset (Ganjoor)
│
├── main.py                    # Main training script (QLoRA)
├── inference.py               # Inference script and text post-processing logic
├── app.py                     # Gradio web UI
├── requirements.txt           # Python dependencies
└── README.md
```

---

## 📊 Dataset

**Source:** Ganjoor Database (Classical Persian Poetry)
*   **Total Poems:** ~54,500
*   **Total Poets:** 67
*   **Data Split:** 90% Train / 10% Validation

---

## ⚙️ Training Configuration

| Parameter | Value | Note |
| :--- | :--- | :--- |
| **Base Model** | `Qwen/Qwen2.5-3B-Instruct` | Multilingual instruction-tuned model |
| **Quantization** | 4-bit (nf4) | To fit into 8GB VRAM |
| **LoRA `r`** | 16 | Rank of LoRA matrices |
| **LoRA `alpha`** | 32 | Scaling factor |
| **Batch Size** | 2 | With `gradient_accumulation_steps=8` |
| **Learning Rate** | 2e-4 | Cosine scheduler |
| **Epochs** | 1 | Sufficient for 50k samples |
| **Max Length** | 256 | Max sequence length |
| **Trainable Params**| ~29.9M (1.73%) | Only Attention layers are trained |


---

## 📈 Evaluation Results

| Metric | Value |
| :--- | :--- |
| **Final Train Loss** | 2.5525 |
| **Final Eval Loss** | 2.5781 |
| **Overfitting** | No (Minimal gap between Train and Eval) |

The model successfully learned the vocabulary and tone of the poets. (Note: Due to the 3B parameter limit, strict adherence to Persian prosody/meter is limited).

---

## ⚠️ Known Limitations

*   **Meter and Rhyme:** 3B parameter models lack deep mathematical understanding of Persian prosody (Aruz). Some verses may lack strict meter.
*   **Length:** The model performs significantly better generating 2 to 4 couplets rather than long poems.
*   **Repetition:** Despite using `repetition_penalty`, the model may occasionally loop words.

---

## 🚀 Usage

### 1. Install Dependencies
Ensure you have Python 3.12 and CUDA installed. Then:
```bash
pip install -r requirements.txt
```
*(Note: The library versions are pinned to ensure compatibility with local 8GB GPUs.)*

### 2. Training (Fine-Tuning)
To start training from scratch or resume from a checkpoint:
```bash
python main.py
```

### 3. Web Application
To run the Gradio UI:
```bash
python app.py
```
Open `http://127.0.0.1:7860` in your browser. Select a poet and click "Generate".

### 4. Command Line Inference
To test the model directly in the terminal:
```bash
python inference.py
```

---

## 🧩 Post-processing Logic
Language models generate continuous text. The `smart_poem_formatter` function in `inference.py` splits the output based on punctuation and word counts, then pairs adjacent segments with 4 spaces to visually reconstruct standard Persian couplets (Beit).

---

## 📜 License
This project is licensed under the MIT License - free to use, modify, and distribute.