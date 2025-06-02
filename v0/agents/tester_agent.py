from agents.base_agent import BaseAgent


class TesterAgent(BaseAgent):
    def act(self, code: str, **kwargs) -> str:
        prompt = f"Write unit tests for the following Python function:\n{code}"
        print(f"[{self.role_name}] Prompting with: {prompt}")
        result = self.generator(prompt, max_new_tokens=4096, truncation=True)[0]["generated_text"]
        return result.strip()

