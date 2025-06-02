import os
import torch

from agents.analyst_agent import AnalystAgent
from agents.coder_agent import CoderAgent
from agents.tester_agent import TesterAgent

ANALYST_PROMPT  = "prompts/analyst_prompt.txt"
ANALYST_OUTPUT  = "outputs/analyst_output.txt"
CODER_PROMPT    = "prompts/coder_prompt.txt"
CODER_OUTPUT    = "outputs/coder_output.txt"
TESTER_PROMPT   = "prompts/tester_prompt.txt"
TESTER_OUTPUT   = "outputs/tester_output.txt"

if torch.cuda.is_available():
    print("CUDA available. Enabling HF offline mode.")
    os.environ["HF_HUB_OFFLINE"] = "1"
else:
    print("CUDA NOT available. Disabling HF offline mode.")
    os.environ["HF_HUB_OFFLINE"] = "0"


os.makedirs("outputs", exist_ok=True)

for path in [ANALYST_OUTPUT, CODER_OUTPUT, TESTER_OUTPUT]:
    open(path, "w").close()

MODEL_NAME = "meta-llama/Llama-3.1-8B"

analyst = AnalystAgent(
    role_name="Analyst",
    model_name=MODEL_NAME,
    device=0,
    prompt_path=ANALYST_PROMPT,
    output_path=ANALYST_OUTPUT
)

coder = CoderAgent(
    role_name="Coder",
    model_name=MODEL_NAME,
    device=1,
    prompt_path=CODER_PROMPT,
    output_path=CODER_OUTPUT
)

tester = TesterAgent(
    role_name="Tester",
    model_name=MODEL_NAME,
    device=2,
    prompt_path=TESTER_PROMPT,
    output_path=TESTER_OUTPUT
)

user_task = "Write a function that calculates the Fibonacci sequence."
with open(ANALYST_PROMPT, "a") as f:
    f.write("\n" + user_task + "\n")

print("\n=== Analyst is generating specification ===")
analyst_output = analyst.act()
print("=== Analyst finished. Spec written to", ANALYST_OUTPUT, "===\n")

with open(ANALYST_OUTPUT, "r") as spec_f:
    spec = spec_f.read().strip()

with open(CODER_PROMPT, "a") as coder_f:
    coder_f.write("\n" + spec + "\n")

print("\n=== Coder is generating code ===")
coder_output = coder.act()
print("=== Coder finished. Code written to", CODER_OUTPUT, "===\n")

with open(CODER_OUTPUT, "r") as code_f:
    code = code_f.read().strip()

with open(TESTER_PROMPT, "a") as test_f:
    test_f.write("\n" + code + "\n")

print("\n=== Tester is generating unit tests ===")
tester_output = tester.act()
print("=== Tester finished. Tests written to", TESTER_OUTPUT, "===\n")

print("Pipeline complete. See outputs in the `outputs/` folder.")


