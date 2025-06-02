from transformers import pipeline


class BaseAgent:
    def __init__(self, role_name, model="meta-llama/Llama-3.1-8B"):
        self.role_name = role_name
        self.generator = pipeline(
                "text-generation",
                model=model,
                device_map="auto",
                torch_dtype="auto",
        )

    def act(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError("Subclasses must implement this.")

