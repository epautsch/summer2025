import os
import torch
from transformers import pipeline
from utils.file_io import read_prompt, read_file, write_file

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

analyst_output = read_file("shared/analyst_output.txt")
prompt = read_prompt("prompts/role_templates/coder.txt", analyst_output)

print("[Coder] Prompting...")
result = generator(prompt, max_new_tokens=1024, truncation=True)[0]["generated_text"]
write_file("shared/coder_output.txt", result)

print("[Coder] Done.")

