from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from PIL import Image
import torch


model_id = "google/gemma-3-27b-it"

model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16
).eval()

processor = AutoProcessor.from_pretrained(model_id)

def save_to_file(text: str, filename: str, mode: str = "a"):
    with open(filename, mode) as f:
        f.write(text.strip() + "\n\n")

def preprocess(messages):
    raw = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
    )

    return {
            "input_ids": raw["input_ids"].to(model.device),
            "attention_mask": raw["attention_mask"].to(model.device),
    }

#################### Analyst part #################################333

system_prompt_analyst = (
        "You are an HPC software analyst. Your role is to take in a problem description "
        "and produce clear specifications for an HPC programmer to code. "
        "Do not write code. Just output a step-by-step spec that a single coder could implement. "
        "Keep the specification under 200 words and in 10 bullet points or less."
)

task_description = (
        "Describe how to implement matrix multiplication in CUDA."
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

inputs_analyst = preprocess(messages_analyst)
input_len_analyst = inputs_analyst["input_ids"].shape[-1]

with torch.inference_mode():
    raw_output_analyst = model.generate(
            **inputs_analyst,
            max_new_tokens=1024,
            do_sample=False,
    )

analyst_ids = raw_output_analyst[0][input_len_analyst:]
analyst_text = processor.decode(analyst_ids, skip_special_tokens=True)

print("=== Analyst says: ===")
print(analyst_text)
print()

output_file = "chain_of_agents_output.txt"
save_to_file("=== Analyst Output ===\n" + analyst_text, output_file, mode="w")

######################## Coder Section ##########################

system_prompt_coder = (
        "You are an HPC coder. You have knowledge of CUDA, SYCL, MPI, OpenMP, and many other "
        "HPC tools. Your role is to take in a software specification from an analyst "
        "and produce clear, working HPC code. Do not include any markdown fences - output only raw code "
        "and in-line comments where appropriate."
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

inputs_coder = preprocess(messages_coder)
input_len_coder = inputs_coder["input_ids"].shape[-1]

with torch.inference_mode():
    raw_output_coder = model.generate(
            **inputs_coder,
            max_new_tokens=2048,
            do_sample=False,
    )

coder_ids = raw_output_coder[0][input_len_coder:]
coder_code = processor.decode(coder_ids, skip_special_tokens=True)

print("=== Coder writes: ===")
print(coder_code)
print()

save_to_file("=== Coder Output ===\n" + coder_code, output_file, mode="a")

############################### Builder (CMake) part ###################################

system_prompt_builder = (
        "You are an HPC build engineer. Your role is to take in working HPC CUDA code from a coder "
        "and produce a CMakeLists.txt file that builds that code correctly. "
        "Do not include any markdown fences or extra explanations - output only the raw contents of CMakeLists.txt."
)

messages_builder = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt_builder}]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": coder_code}]
        }
]

inputs_builder = preprocess(messages_builder)
input_len_builder = inputs_builder["input_ids"].shape[-1]

with torch.inference_mode():
    raw_output_builder = model.generate(
            **inputs_builder,
            max_new_tokens=512,
            do_sample=False,
    )

builder_ids = raw_output_builder[0][input_len_builder:]
builder_code = processor.decode(builder_ids, skip_special_tokens=True)

print("=== Builder generates CMakeLists.txt: ===")
print(builder_code)
print()

save_to_file("=== Builder Output ===\n" + builder_code, output_file, mode="a")

############################## tester part #####################################

system_prompt_tester = (
        "You are an HPC code tester. Your job is to take in working HPC code "
        "and generate appropriate unit tests using the appropriate framework. "
        "Do not include any markdown fences or extra explanations - output only the raw test code "
        "and in-line comments if needed."
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

inputs_tester = preprocess(messages_tester)
input_len_tester = inputs_tester["input_ids"].shape[-1]

with torch.inference_mode():
    raw_output_tester = model.generate(
            **inputs_tester,
            max_new_tokens=2048,
            do_sample=False,
    )

tester_ids = raw_output_tester[0][input_len_tester:]
tester_tests = processor.decode(tester_ids, skip_special_tokens=True)

print("=== Tester generates unit tests: ===")
print(tester_tests)
print()

save_to_file("=== Tester Output ===\n" + tester_tests, output_file, mode="a")

