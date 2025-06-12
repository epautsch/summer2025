import json
import subprocess
import os
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import torch
import re


YELLOW = "\033[33m"
GREEN = "\033[32m"
BLUE = "\033[34m"
RESET = "\033[0m"

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
    print(f"[DEBUG] Saving to {filename}")
    with open(filename, "w") as f:
        f.write(text.strip() + "\n")

def maybe_summarize(history: str, original_task: str, word_limit: int = 400) -> str:
    words = history.split()
    if len(words) <= word_limit:
        print(f"[DEBUG] Word limit is {len(words)}... NOT calling summarizer.")
        return history
    else:
        print(f"[DEBUG] Word limit is {len(words)}... calling summarizer.")

    prompt = (
        f"Original task: {original_task}\n\n"
        "Conversation history:\n"
        f"{history}\n\n"
        "Please summarize the above conversation history in 50-100 words, "
        "preserving the original task exactly as given."
    )

    summary = summarizer.generate(prompt)

    new_history = (
        f"Original task: {original_task}\n"
        "History summary:\n"
        f"{summary}\n"
    )
    print("[DEBUG] History was long; ran summarizer →")
    print(summary, "\n")
    return new_history

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
        "You are an HPC coder. Take a specification and produce working code files "
        "(e.g., .cu, .cpp, .h) with inline comments. Ouput raw code only."
    ),
    max_new_tokens=1024
)

summarizer = Agent(
    role_name="Summarizer",
    system_prompt=(
        "You are a concise summarizer. Given a long conversation history, produce a 50-100 word summary "
        "that captures thought/action/observation and reatains the original task context."
    ),
    max_new_tokens=256
)

################### LLM interaction helpers ########################

def preprocess_for_llm(history: str, agent_prompt: str, max_new_tokens: int):
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are an HPC manager agent. "
            "When given a 'Thought:' prompt, you must reply in JSON with exactly two fields: "
            "\"thought\" and \"action\". Valid actions are:\n"
            "1) call_analyst\n"
            "2) call_coder\n"
            "3) compile_code\n"
            "4) run_binary\n"
            "5) system_command\n"
            "6) finish\n"
            "If the action is call_analyst or call_coder, you must supply a \"payload\" string "
            "(the input to that sub-agent). If action is compile_code, run_binary, or system_command, "
            "the payload should be the shell command. If finish, payload should be the final summary. "
            "The typical workflow should be to use code_analyst to get a code specification, hand "
            "the code specification to the coder with call_coder, use the compile_code action to compile "
            "code, and use run_binary to check if the executable works. You can use system_command as the "
            "action if you need to make system calls to get more information or make verifications about "
            "the system. You can use finish to signal that you have completed the task. Do not assume that "
            "the analyst or coder can open files. You need to pass messages and code directly."
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
    #print(f"[DEBUG] Raw LLM response:\n{decoded}\n")
    return decoded.strip()

def parse_manager_response(resp: str) -> dict:
    #print(f"[DEBUG] Stripping fences from response...")
    stripped = strip_markdown_fences(resp)
    #print(f"[DEBUG] After stripping fences:\n{stripped}\n")
    try:
        data = json.loads(stripped)
        pretty = json.dumps(data, indent=2)
        #print("[DEBUG] Parsed JSON:")
        #print(pretty + "\n")
        if all(k in data for k in ("thought", "action", "payload")):
            return data
        else:
            print(f"[DEBUG] Missing keys in JSON, found keys: {list(data.keys())}")
    except json.JSONDecodeError as e:
        print(f"[DEBUG] JSON decode error: {e}")
    raise ValueError(f"Invalid JSON from manager: {stripped}")

############## manager's react loop ###################

def main_react():
    history = ""
    original_task = "Implement basic matrix multiplication in CUDA."
    agent_prompt = f"Thought: The user requested: \"{original_task}\". What should be my first action?"

    for i in range(20):
        print(f"\n=========== Loop iteration {i} ===========")

        history = maybe_summarize(history, original_task, word_limit=300)
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
        
        print(f"{YELLOW}Thought: {thought}{RESET}")
        print(f"{GREEN}Action: {action}{RESET}")
        print(f"{BLUE}Payload: {json.dumps(payload, indent=2) if isinstance(payload, dict) else payload}\n{RESET}")

        history += f"Thought: {thought}\nAction: {action} | Payload: {payload}\n"

        if action == "call_analyst":
            inp = payload["input"] if isinstance(payload, dict) else payload
            fname = payload.get("filename", "analyst_spec.txt") if isinstance(payload, dict) else "analyst_spec.txt"
            spec = analyst.generate(inp)
            save_to_file(spec, fname)
            observation = f"Analyst output saved to {fname}"

        elif action == "call_coder":
            inp = payload["input"] if isinstance(payload, dict) else payload
            fname = payload.get("filename", "code_output.cu") if isinstance(payload, dict) else "code_output.cu"
            code = coder.generate(inp)
            save_to_file(code, fname)
            observation = f"Coder output saved to {fname}"

        elif action == "compile_code":
            cmd = payload
            out = run_shell(cmd)
            observation = f"Compile output:\n{out}"

        elif action == "run_binary":
            cmd = payload
            out = run_shell(cmd)
            observation = f"Run output:\n{out}"

        elif action == "system_command":
            cmd = payload
            out = run_shell(cmd)
            observation = f"System command output:\n{out}"

        elif action == "finish":
            print("=== MANAGER FINAL SUMMARY ===")
            print(payload)
            return

        else:
            observation = f"Unknown action: {action}"
        
        print(f"[DEBUG] Observation: {observation}")
        history += f"Observation: {observation}\n"
        agent_prompt = "Thought: Given the above observation, what is the next action?"

    print("Reached maximum iteration limit without 'finish' action.")

if __name__ == "__main__":
    main_react()
    
