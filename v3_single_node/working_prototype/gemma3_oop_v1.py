from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import torch
import re


############## model setup ########################

model_id = "google/gemma-3-27b-it"

model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16
).eval()

processor = AutoProcessor.from_pretrained(model_id)

####################### helper funcs ##########################

def strip_markdown_fences(text: str) -> str:
    lines = text.strip().splitlines()
    if len(lines) >= 2 and re.match(r"^```", lines[0]) and re.match(r"^```", lines[-1]):
        return "\n".join(lines[1:-1]).strip()
    return text.strip()

def save_to_file(text: str, filename: str, mode: str = "w"):
    with open(filename, mode) as f:
        f.write(text.strip() + "\n")

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

class Agent:
    def __init__(self, role_name: str, system_prompt: str, max_new_tokens: int, output_filename: str):
        self.role_name = role_name
        self.system_prompt = system_prompt
        self.max_new_tokens = max_new_tokens
        self.output_filename = output_filename

    def generate(self, user_input: str) -> str:
        messages = [
                {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
                {"role": "user", "content": [{"type": "text", "text": user_input}]},
        ]

        inputs = preprocess(messages)
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            raw_out = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
            )

        generated_ids = raw_out[0][input_len:]
        decoded = processor.decode(generated_ids, skip_special_tokens=True)
        return strip_markdown_fences(decoded)

    def save(self, output_text: str):
        save_to_file(output_text, self.output_filename)

############################# Agent setups #########################################

analyst = Agent(
        role_name="Analyst",
        system_prompt=(
            "You are an HPC software analyst. Your role is to take in a problem description "
            "and produce clear specifications for an HPC programmer to code. "
            "Do not write code. Just output a step-by-step spec that a single coder could implement. "
            "Keep the specification under 200 words and in 10 bullet points or less."
        ),
        max_new_tokens=1024,
        output_filename="analyst_output.txt"
)

coder = Agent(
        role_name="Coder",
        system_prompt=(
            "You are an HPC coder. You have knowledge of CUDA, SYCL, MPI, OpenMP, and many other "
            "HPC tools. Your role is to take in a software specification from an analyst "
            "and produce clear, working HPC code. Do not include any markdown fences - output only raw code "
            "and in-line comments where appropriate."
        ),
        max_new_tokens=2048,
        output_filename="coder_output.cu"
)

builder = Agent(
        role_name="Builder",
        system_prompt=(
            "You are an HPC build engineer. Your role is to take in working HPC CUDA code from a coder "
            "and produce a CMakeLists.txt file that builds that code correctly. "
            "Do not include any markdown fences or extra explanations - output only the raw contents of "
            "CMakeLists.txt."
        ),
        max_new_tokens=512,
        output_filename="CMakeLists.txt"
)

tester = Agent(
        role_name="Tester",
        system_prompt=(
            "You are an HPC code tester. Your job is to take in working HPC code "
            "and generate appropriate unit tests using the appropriate framework. "
            "Do not include any markdown fences or extra explanations - output only the raw test code "
            "and in-line comments if needed."
        ),
        max_new_tokens=2048,
        output_filename="tester_output.cpp"
)

########################### Main ####################################

if __name__ == "__main__":

    task_description = "Describe how to implement matrix multiplication in CUDA."
    analyst_text = analyst.generate(task_description)
    print("=== Analyst says: ===")
    print(analyst_text + "\n")
    analyst.save(analyst_text)

    coder_text = coder.generate(analyst_text)
    print("=== Coder writes: ===")
    print(coder_text + "\n")
    coder.save(coder_text)

    builder_text = builder.generate(coder_text)
    print("=== Builder generates CMakeLists.txt: ===")
    print(builder_text + "\n")
    builder.save(builder_text)

    tester_text = tester.generate(coder_text)
    print("=== Tester generates unit tests: ===")
    print(tester_text + "\n")
    tester.save(tester_text)

