import gradio as gr
from inference import load_model_for_inference, generate_poem

# =====================================================
# 1. List of poets (from the most frequent poets in the dataset)
# =====================================================
POETS = [
    "صائب تبریزی",
    "مولوی",
    "عطار",
    "بیدل دهلوی",
    "شاه نعمت‌الله ولی",
    "سعدی",
    "سنایی",
    "انوری",
    "خاقانی",
    "اوحدی",
    "ملک‌الشعرای بهار",
    "فیض کاشانی",
    "محتشم کاشانی",
    "اقبال لاهوری",
    "خواجوی کرمانی",
    "سلمان ساوجی",
    "امیرخسرو دهلوی",
    "ابوسعید ابوالخیر",
    "عرفی",
    "وحشی",
    "قاآنی",
    "فردوسی",
    "حافظ",
    "عراقی",
    "فروغی بسطامی",
    "رودکی",
    "ناصرخسرو",
    "نظامی",
    "باباطاهر",
]

# =====================================================
# 2. CSS style to align the entire page to the right (RTL)
# =====================================================
RTL_CSS = """
body, .gradio-container {
    direction: rtl !important;
    text-align: right !important;
}

input, textarea, select, .gr-box {
    direction: rtl !important;
    text-align: right !important;
}

.prose {
    text-align: right !important;
}
"""

# =====================================================
# 3. Load the model (only once, at app startup)
# =====================================================
print("⏳ Loading model... (This may take a few minutes)")
model, tokenizer = load_model_for_inference()
print("✅ Model loaded successfully!")


# =====================================================
# 4. Poetry generation function (connecting to the UI)
# =====================================================
def predict(poet_name):
    """Function bridging the UI and the model."""
    if not poet_name:
        return "لطفاً یک شاعر را از لیست انتخاب کنید."

    try:
        poem = generate_poem(model, tokenizer, poet_name)
        if not poem or len(poem.strip()) == 0:
            return "Unfortunately, the model could not generate a poem. Please try again."
        return poem
    except Exception as e:
        return f"❌ An error occurred: {str(e)}"


# =====================================================
# 5. UI design with Gradio
# =====================================================
# Injecting CSS into Blocks
with gr.Blocks(theme=gr.themes.Soft(), title="تولیدکننده شعر فارسی", css=RTL_CSS) as demo:

    gr.Markdown("""
        # 🌹 تولیدکننده شعر فارسی
        ### با استفاده از مدل <span dir="ltr">**Qwen2.5-3B**</span> و تکنیک <span dir="ltr">**QLoRA**</span>
        
        نام شاعر مورد نظر خود را از لیست انتخاب کنید و روی دکمه **«ساخت شعر»** بزنید تا مدل در سبک او شعر بسراید.
        """)

    with gr.Row():
        with gr.Column(scale=1):
            poet_dropdown = gr.Dropdown(
                choices=POETS,
                value="مولوی",
                label="🖋️ شاعر مورد نظر",
                info="از بین شاعران آموزش‌دیده انتخاب کنید",
            )

            generate_btn = gr.Button(
                "✨ ساخت شعر",
                variant="primary",
                size="lg",
            )

            gr.Markdown("""
                ---
                **💡 نکته:**
                چون مدل روی کارت گرافیک محلی آموزش دیده، تولید هر شعر ممکن است ۱۰ تا ۳۰ ثانیه طول بکشد.
                """)

        with gr.Column(scale=2):
            output_text = gr.Textbox(
                label="📜 شعر تولید شده",
                placeholder="شعر شما اینجا نمایش داده می‌شود...",
                lines=15,
                max_lines=20,
                show_copy_button=True,
                rtl=True,
            )

    # Connecting the button to the function
    generate_btn.click(
        fn=predict,
        inputs=[poet_dropdown],
        outputs=[output_text],
    )


# =====================================================
# 6. Running the app
# =====================================================
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
