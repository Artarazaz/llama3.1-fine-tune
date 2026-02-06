import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cpu"

MODEL_PATH = "/notebooks/fine-tuned-model-2/best"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(device)
model.eval()


def generate_text(prompt, max_new_tokens=30, temperature=0.7):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # تولید متن امن
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


prompt = "به پارتنرم خیانت کردم. باید ناراحت باشم راجبش؟ پاسخ کوتاه بده"
output = generate_text(prompt)
print("خروجی مدل:", output)