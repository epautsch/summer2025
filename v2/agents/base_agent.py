from transformers import pipeline
import os


class BaseAgent:
    def __init__(self,
                 role_name: str,
                 role_description: str,
                 model_name: str,
                 device: int,
                 write_path: str):
        self.role_name = role_name
        self.role_description = role_description
        self.model_name = model_name
        self.device = device
        self.write_path = write_path

        if os.environ.get("HF_HUB_OFFLINE") is None:
            os.environ["HF_HUB_OFFLINE"] = "0"

        self.generator = pipeline(
            "text-generation",
            model=self.model_name,
            device=self.device,
            torch_dtype="auto",
            return_full_text=False
        )

        self.additional_info = ""
        self.output = ""

    def receive_additional_info(self, add_info_path: str):
        with open(add_info_path, "r") as f:
            self.additional_info = f.read()

    def infer(self):
        print(f"**{self.role_name}** is now inferencing...")

        sys_part = f"System: {self.role_description.strip()}"
        user_part = f"User: {self.additional_info.strip()}"

        inference_input = sys_part + "\n\n" + user_part

        self.output = self.generator(
                        inference_input,
                        min_new_tokens=100,
                        max_new_tokens=300,
                        truncation=True,
        )[0]["generated_text"].strip()

    def write_out(self):
        with open(self.write_path, "w") as f:
            f.write(self.output.strip())

