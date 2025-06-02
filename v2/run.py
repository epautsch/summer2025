import os
import torch

from agents.base_agent import BaseAgent


if torch.cuda.is_available():
    print("CUDA available. Enabling HF offline mode.")
    os.environ["HF_HUB_OFFLINE"] = "1"
else:
    print("CUDA NOT available. Disabling HF offline mode.")
    os.environ["HF_HUB_OFFLINE"] = "0"

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

analyst_role_description = (
        "You are a software analyst. Your role is to take in a problem description "
        "and produce clear specifications for a Python programmer to code. "
        "Do not write code. Just output a step-by-step spec that a single coder could implement. "
        "Keep the specification under 200 words and in no more than 5 bullet points."
)

analyst = BaseAgent(
        role_name="Analyst",
        role_description=analyst_role_description,
        model_name=MODEL_NAME,
        device=0,
        write_path="outputs/analyst_output.txt"
)

analyst.receive_additional_info("prompts/analyst_prompt.txt")
analyst.infer()
analyst.write_out()

coder_role_description = (
        "You are a Python coder. Your role is to take in a software specification from an analyst "
        "and produce clear Python code. "
        "Do not write anything else but Python code and in-line comments where appropriate."
)

coder = BaseAgent(
        role_name="Coder",
        role_description=coder_role_description,
        model_name=MODEL_NAME,
        device=1,
        write_path="outputs/coder_output.txt"
)

coder.receive_additional_info("outputs/analyst_output.txt")
coder.infer()
coder.write_out()

