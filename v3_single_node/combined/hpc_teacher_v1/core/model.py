model_id = "google/gemma-3-27b-it"
llm_model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id,
    attn_implementation="eager",
    device_map="auto",
    torch_dtype=torch.bfloat16
).eval()
processor=AutoProcessor.from_pretrained(model_id)

@dataclass
class ManagerModel:
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
        console.print(f"[red][DEBUG PAYLOAD][/]\n{payload}")
        console.print(f"[red][DEBUG PAYLOAD END][/]\n")

        with console.status("Generating response...", spinner="dots"):
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
                    #cache_implementation="offloaded",
                    do_sample=False
                )
            gen_ids = out[0][input_len:]
            decoded = self.processor.decode(gen_ids, skip_special_tokens=True)
            console.print("[red][DEBUG MANAGERMODEL GEN][/]")
            console.print(decoded)
            console.print("[red][DEBUG MANAGERMODEL GEN END][/]\n")
            return decoded


