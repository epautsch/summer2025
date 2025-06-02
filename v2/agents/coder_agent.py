from agents.base_agent import BaseAgent


class CoderAgent(BaseAgent):
    def __init__(self,
                 role_name: str,
                 model_name: str,
                 device: int,
                 prompt_path: str,
                 output_path: str):
        super().__init__(role_name, model_name, device)
        self.prompt_path = prompt_path
        self.output_path = output_path

    def act(self) -> str:
        with open(self.prompt_path, "r") as f:
            prompt_text = f.read()

        print(f"[{self.role_name}] Running on device {self.generator.device} with prompt:")
        print(prompt_text)
        gen = self.generator(
                prompt_text,
                max_new_tokens=2048,
                truncation=True
        )[0]["generated_text"]

        with open(self.output_path, "a") as out_f:
            out_f.write("\n" + gen.strip() + "\n")
        return gen.strip()

