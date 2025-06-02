from agents.base_agent import BaseAgent


class CoderAgent(BaseAgent):
    def act(self, prompt: str, **kwargs) -> str:
        print(f"[{self.role_name}] Prompting with: {prompt}")
        result = self.generator(prompt, max_new_tokens=4096, truncation=True)[0]["generated_text"]
        return result.strip()

