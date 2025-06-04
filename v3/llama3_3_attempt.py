import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


model_id = "meta-llama/Llama-3.3-70B-Instruct"
quantization_config = BitsAndBytesConfig(load_in_8bit=True)

quantized_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
input_text = "This is a test. Are you working?"
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

output = quantized_model.generate(**input_ids, use_cache=False, max_new_tokens=64)

print(tokenizer.decode(output[0], skip_special_tokens=True))

