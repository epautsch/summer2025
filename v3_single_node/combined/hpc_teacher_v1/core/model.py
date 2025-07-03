from dataclasses import dataclass
from typing import Any

import torch
from rich.console import Console

console = Console()


@dataclass
class LLMClient:
    model: Any
    processor: Any
    system_prompt: str
    max_new_tokens: int

    def generate(self, user_prompt: str) -> str:
        payload = [
            {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": user_prompt}]}
        ]
        # DEBUG print
        console.print("[blue]▶️ LLMClient.generate() payload:[/]\n", payload)

        with console.status("Generating response...", spinner="dots"):
            # tokenize & run
            raw = self.processor.apply_chat_template(
                payload,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.model.device, dtype=torch.bfloat16)

            input_len = raw["input_ids"].shape[-1]
            with torch.inference_mode():
                out = self.model.generate(
                    **raw,
                    max_new_tokens=self.max_new_tokens,
                    # cache_implementation="offloaded",
                    do_sample=False
                )
            # decode
            gen_ids = out[0][input_len:]
            decoded = self.processor.decode(gen_ids, skip_special_tokens=True)
            return decoded
