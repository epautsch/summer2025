import os
import torch
from transformers import pipeline
from utils.file_io import read_prompt, write_file


if torch.cuda.is_available():
    print("CUDA available. Offline mode enabled.")
    os.environ["HF_HUB_OFFLINE"] = "1"
else:
    print("No CUDA. Login node mode.")
    os.environ["HF_HUB_OFFLINE"] = "0"

model_name = "meta-llama/Llama-3.1-8B"
generator = pipeline(
        "text-generation",
        model=model_name,
        device_map="auto",
        torch_dtype="auto",
)

task = "Write a Python function that calculates the Fibonacci sequence up to a given number."

prompt = read_prompt("prompts/role_templates/analyst.txt", task)

print("[Analyst] Prompting...")
result = generator(prompt, max_new_tokens=1024, truncation=True)[0]["generated_text"]
write_file("shared/analyst_output.txt", result)

print("[Analyst] Done.")

