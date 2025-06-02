def read_prompt(prompt_file, extra_context=""):
    with open(prompt_file, "r") as f:
        base_prompt = f.read()
    return f"{base_prompt}\n\n{extra_context.strip()}"

def read_file(filepath):
    with open(filepath, "r") as f:
        return f.read().strip()

def write_file(filepath, content):
    with open(filepath, "w") as f:
        f.write(content.strip())

