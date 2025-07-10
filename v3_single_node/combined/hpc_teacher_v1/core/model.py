from dataclasses import dataclass
from typing import Any

import torch
from rich.console import Console

console = Console()


@dataclass
class LLMClient:
    hf_model: Any
    processor: Any
    system_prompt: str
    max_new_tokens: int

    def generate(self, user_prompt: str) -> str:
        input = [
            {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": user_prompt}]}
        ]
        # DEBUG print
        #console.print("[blue]▶️  LLMClient.generate() input:[/]\n", input)

        with console.status("Generating response...", spinner="dots"):
            # tokenize & run
            raw = self.processor.apply_chat_template(
                input,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.hf_model.device, dtype=torch.bfloat16)
           # )

            # new for 1b model
            #for k, v in raw.items():
            #    if isinstance(v, torch.Tensor):
            #        raw[k] = v.to(self.model.device, dtype=torch.bfloat16)

            input_len = raw["input_ids"].shape[-1]
            with torch.inference_mode():
                out = self.hf_model.generate(
                    **raw,
                    max_new_tokens=self.max_new_tokens,
                    # cache_implementation="offloaded",
                    do_sample=False
                )
            # decode
            gen_ids = out[0][input_len:]
            decoded = self.processor.decode(gen_ids, skip_special_tokens=True)
            #console.print("[red]▶️  LLMClient.generate() output:[/]\n", decoded)
            return decoded
