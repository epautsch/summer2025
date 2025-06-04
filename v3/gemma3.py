from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from PIL import Image
import torch


model_id = "google/gemma-3-27b-it"

model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
).eval()

processor = AutoProcessor.from_pretrained(model_id)

#################### Analyst part #################################333

system_prompt_analyst = (
        "You are a software analyst. Your role is to take in a problem description "
        "and produce clear specifications for a Python programmer to code. "
        "Do not write code. Just output a step-by-step spec that a single coder could implement. "
        "Keep the specification under 200 words and in no more than 5 bullet points."
)

task_description = (
        "Describe how to implement a function that calculates the Fibonacci sequence up to a given number."
)

messages_analyst = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt_analyst}]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": task_description}]
        }
]

inputs_analyst = processor.apply_chat_template(
        messages_analyst,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
).to(model.device, dtype=torch.bfloat16)

input_len_analyst = inputs_analyst["input_ids"].shape[-1]

with torch.inference_mode():
    raw_output_analyst = model.generate(
            **inputs_analyst,
            max_new_tokens=512,
            do_sample=False,
    )
    analyst_ids = raw_output_analyst[0][input_len_analyst:]

analyst_text = processor.decode(analyst_ids, skip_special_tokens=True)

print("=== Analyst says: ===")
print(analyst_text)
print()

######################## Coder Section ##########################

system_prompt_coder = (
        "You are a Python coder. Your role is to take in a software specification from an analyst "
        "and produce clear, working Python code. Do not write anything else but Python code and "
        "in-line comments where appropriate."
)

messages_coder = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt_coder}]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": analyst_text}]
        }
]

inputs_coder = processor.apply_chat_template(
        messages_coder,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
).to(model.device, dtype=torch.bfloat16)

input_len_coder = inputs_coder["input_ids"].shape[-1]

with torch.inference_mode():
    raw_output_coder = model.generate(
            **inputs_coder,
            max_new_tokens=512,
            do_sample=False,
    )
    coder_ids = raw_output_coder[0][input_len_coder:]

coder_code = processor.decode(coder_ids, skip_special_tokens=True)

print("=== Coder writes: ===")
print(coder_code)
print()

############################## tester part #####################################

system_prompt_tester = (
        "You are a Python tester. Your job is to take in working Python code "
        "and generate appropriate unit tests using the unittest framework. "
        "Do not write anything else - only produce test code and in-line comments if needed."
)

messages_tester = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt_tester}]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": coder_code}]
        }
]

inputs_tester = processor.apply_chat_template(
        messages_tester,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
).to(model.device, dtype=torch.bfloat16)

input_len_tester = inputs_tester["input_ids"].shape[-1]

with torch.inference_mode():
    raw_output_tester = model.generate(
            **inputs_tester,
            max_new_tokens=512,
            do_sample=False,
    )
    tester_ids = raw_output_tester[0][input_len_tester:]

tester_tests = processor.decode(tester_ids, skip_special_tokens=True)

print("=== Tester generates unit tests: ===")
print(tester_tests)

