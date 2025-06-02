from agents.base_agent import BaseAgent


class AnalystAgent(BaseAgent):
    def act(self, task_description: str, **kwargs) -> str:
        prompt = f"Analyze the following task and write a clear specification for a programmer:\n{task_description}"
        print(f"[{self.role_name}] Prompting with: {prompt}")
        result = self.generator(prompt, max_new_tokens=4096, truncation=True)[0]["generated_text"]
        return result.strip()

