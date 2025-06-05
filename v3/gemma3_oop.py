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

class Agent:
    def __init__(self, role_name: str, system_prompt: str, max_new_tokens: int):
        self.role_name = role_name
        self.system_prompt = system_prompt
        self.max_new_tokens = max_new_tokens

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
        generated_text = processor.decode(generated_ids, skip_special_tokens=True)
        return generated_text

    def save(self, output_text: str, filename: str, overwrite: bool = False):
        mode = "w" if overwrite else "a"
        header = f"=== {self.role_name} Output ===\n"
        save_to_file(header + output_text, filename, mode=mode)

############################# Agent setups #########################################

analyst = Agent(
        role_name="Analyst",
        system_prompt=(
            "You are an HPC software analyst. Your role is to take in a problem description "
            "and produce clear specifications for an HPC programmer to code. "
            "Do not write code. Just output a step-by-step spec that a single coder could implement. "
            "Keep the specification under 200 words and in 10 bullet points or less."
        ),
        max_new_tokens=1024
)

coder = Agent(
        role_name="Coder",
        system_prompt=(
            "You are an HPC coder. You have knowledge of CUDA, SYCL, MPI, OpenMP, and many other "
            "HPC tools. Your role is to take in a software specification from an analyst "
            "and produce clear, working HPC code. Do not include any markdown fences - output only raw code "
            "and in-line comments where appropriate."
        ),
        max_new_tokens=2048
)

builder = Agent(
        role_name="Builder",
        system_prompt=(
            "You are an HPC build engineer. Your role is to take in working HPC CUDA code from a coder "
            "and produce a CMakeLists.txt file that builds that code correctly. "
            "Do not include any markdown fences or extra explanations - output only the raw contents of "
            "CMakeLists.txt."
        ),
        max_new_tokens=512
)

tester = Agent(
        role_name="Tester",
        system_prompt=(
            "You are an HPC code tester. Your job is to take in working HPC code "
            "and generate appropriate unit tests using the appropriate framework. "
            "Do not include any markdown fences or extra explanations - output only the raw test code "
            "and in-line comments if needed."
        ),
        max_new_tokens=2048
)

########################### Main ####################################

if __name__ == "__main__":
    output_file = "chain_of_agents_output.txt"

    task_description = "Describe how to implement matrix multiplication in CUDA."
    analyst_text = analyst.generate(task_description)
    print("=== Analyst says: ===")
    print(analyst_text + "\n")
    analyst.save(analyst_text, output_file, overwrite=True)

    coder_text = coder.generate(analyst_text)
    print("=== Coder writes: ===")
    print(coder_text + "\n")
    coder.save(coder_text, output_file)

    builder_text = builder.generate(coder_text)
    print("=== Builder generates CMakeLists.txt: ===")
    print(builder_text + "\n")
    builder.save(builder_text, output_file)

    tester_text = tester.generate(coder_text)
    print("=== Tester generates unit tests: ===")
    print(tester_text + "\n")
    tester.save(tester_text, output_file)

