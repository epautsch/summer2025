import json
import subprocess
import os
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import torch
import re


###### model and processor ########

model_id = "google/gemma-3-27b-it"
model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16
).eval()
processor = AutoProcessor.from_pretrained(model_id)

############## utility funcs #################

def strip_markdown_fences(text: str) -> str:
    lines = text.strip().splitlines()
    if len(lines) >= 2 and re.match(r"^```", lines[0]) and re.match(r"^```", lines[-1]):
        return "\n".join(lines[1:-1]).strip()
    return text.strip()

def run_shell(cmd: str) -> str:
    print(f"DEBUG] Running shell command: {cmd}")
    proc = subprocess.Popen(
        cmd, shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    out, _ = proc.communicate()
    decoded = out.decode("utf-8", errors="ignore")
    print(f"[DEBUG] Shell output:\n{decoded}")
    return decoded

def save_to_file(text: str, filename: str):
    print(f"[DEBUG] Saving to {filename} (overwrite).")
    with open(filename, "w") as f:
        f.write(text.strip() + "\n")

def preprocess_for_llm(history: str, agent_prompt: str, max_new_tokens: int):
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are an HPC manager agent. "
            "When given a 'Thought:' prompt, you must reply in JSON with exactly two fields: "
            "\"thought\" and \"action\". Valid actions are :\n"
            "1) call_analyst\n"
            "2) call_coder\n"
            "3) compile_code\n"
            "4) run_binary\n"
            "5) finish\n"
            "If the action is call_analyst or call_coder, you must supply a \"payload\" string "
            "(the input to that sub-agent). If action is compile_code or run_binary, the payload "
            "should be the shell command. If finish, payload should be the final summary."
        }]},
        {"role": "user", "content": [{"type": "text", "text": history + "\n" + agent_prompt}]}
    ]

    raw = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)

    input_len = raw["input_ids"].shape[-1]
    with torch.inference_mode():
        gen = model.generate(
            **raw, max_new_tokens=max_new_tokens, do_sample=False
        )
    output_ids = gen[0][input_len:]
    decoded = processor.decode(output_ids, skip_special_tokens=True)
    print(f"[DEBUG] Raw LLM response:\n{decoded}\n")
    return decoded.strip()

def parse_manager_response(resp: str) -> dict:
    print(f"[DEBUG] Stripping fences from response...")
    stripped = strip_markdown_fences(resp)
    print(f"[DEBUG] After stripping fences:\n{stripped}\n")
    try:
        data = json.loads(stripped)
        pretty = json.dumps(data, indent=2)
        print("[DEBUG] Parsed JSON:")
        print(pretty + "\n")
        if all(k in data for k in ("thought", "action", "payload")):
            return data
        else:
            print(f"[DEBUG] Missing keys in JSON, found keys: {list(data.keys())}")
    except json.JSONDecodeError as e:
        print(f"[DEBUG] JSON decode error: {e}")
    raise ValueError(f"Invalid JSON from manager: {stripped}")

############### Agent class and sub-agents #################

class Agent:
    def __init__(self, role_name: str, system_prompt: str, max_new_tokens: int):
        self.role_name = role_name
        self.system_prompt = system_prompt
        self.max_new_tokens = max_new_tokens

    def generate(self, user_input: str) -> str:
        print(f"[DEBUG] {self.role_name}.generate() input:\n{user_input}\n")
        messages = [
            {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": user_input}]},
        ]
        raw = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(model.device, dtype=torch.bfloat16)

        input_len = raw["input_ids"].shape[-1]
        with torch.inference_mode():
            out = model.generate(
                **raw, max_new_tokens=self.max_new_tokens, do_sample=False
            )
        gen_ids = out[0][input_len:]
        decoded = processor.decode(gen_ids, skip_special_tokens=True)
        clean = strip_markdown_fences(decoded)
        print(f"[DEBUG] {self.role_name}.generate() output:\n{clean}\n")
        return clean

analyst = Agent(
    role_name="Analyst",
    system_prompt=(
        "You are an HPC software analyst. Take a problem description and produce a software spec. "
        "Keep it under 200 words and in 10 bullet points or fewer. Do not write any code."
    ),
    max_new_tokens=1024
)

coder = Agent(
    role_name="Coder",
    system_prompt=(
        "You are an HPC CUDA coder. Take a specification and produce complete .cu code, "
        "with inline comments. Do not output markdown fences—only raw code."
    ),
    max_new_tokens=2048
)

############## manager's react loop ###################

def main_react():
    history = ""
    task = "User wants to do matrix multiplication in CUDA."
    agent_prompt = f"Thought: The user requested: \"{task}\". What should be my first action?"

    for i in range(10):
        print(f"\n=========== Loop iteration {i} ===========")
        resp_text = preprocess_for_llm(history, agent_prompt, max_new_tokens=1024)

        try:
            mgr = parse_manager_response(resp_text)
        except ValueError as e:
            err = str(e)
            print(f"[DEBUG] Manager parse failure: {err}")
            history += f"Observation: Manager JSON parse failed: {err}\n"
            agent_prompt = (
                "Thought: My last JSON was invalid. "
                "I need to output a JSON object with exactly "
                "\"thought\", \"action\", and \"payload\" fields—no extra text.\n"
                "Action: retry_json\n"
                "Payload: none\n\n"
                "Now, produce only the corrected JSON."
            )
            continue

        thought, action, payload = mgr["thought"], mgr["action"], mgr["payload"]
        print("[DEBUG] Manager decisions:")
        print(json.dumps({
            "thought": thought,
            "action": action,
            "payload": payload
        }, indent=2) + "\n")

        history += f"Thought: {thought}\nAction: {action} | Payload: {payload}\n"

        if action == "call_analyst":
            spec = analyst.generate(payload)
            save_to_file(spec, "analyst_spec.txt")
            observation = f"Analyst output saved to analyst_spec.txt:\n{spec}"
        elif action == "call_coder":
            if os.path.exists(payload):
                with open(payload) as f:
                    spec_text = f.read()
            else:
                spec_text = payload
            code = coder.generate(spec_text)
            save_to_file(code, "matmul.cu")
            observation = f"Coder output saved to matmul.cu:\n{code[:200]}...\n"
        elif action == "compile_code":
            out = run_shell(payload)
            observation = f"Compile output:\n{out}"
        elif action == "run_binary":
            out = run_shell(payload)
            observation = f"Run output:\n{out}"
        elif action == "finish":
            summary = payload
            print("=== MANAGER FINAL SUMMARY ===")
            print(summary)
            return
        else:
            observation = f"Unknown action: {action}"
        
        print(f"[DEBUG] Observation: {observation}")
        history += f"Observation: {observation}\n"
        agent_prompt = "Thought: Given the above observation, what is the next action?"

    print("Reached maximum iteration limit without 'finish' action.")

if __name__ == "__main__":
    main_react()
    
