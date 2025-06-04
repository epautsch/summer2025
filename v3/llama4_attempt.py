#from transformers import AutoTokenizer, LlamaForCausalLM
from transformers import AutoProcessor, Llama4ForConditionalGeneration
import torch

#model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-4-Scout-17B-16E-Instruct")
#tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-4-Scout-17B-16E-Instruct")

#prompt = "Hey, are you working? This is a test."
#inputs = tokenizer(prompt, return_tensors="pt")

#generate_ids = model.generate(inputs.input_ids, max_length=30)
#tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
#model_id = "RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic"

processor = AutoProcessor.from_pretrained(model_id)
model = Llama4ForConditionalGeneration.from_pretrained(
        model_id,
        attn_implementation="flex_attention",
        device_map="auto",
        torch_dtype=torch.bfloat16,
)

messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "This is a test. Are you working?"},
            ]
        },
]

inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
).to(model.device)

outputs = model.generate(
        **inputs,
        max_new_tokens=20,
)

response = processor.batch_decode(outputs[:, inputs["input_ids"].shape[-1]:])[0]
print(response)
print(outputs[0])

