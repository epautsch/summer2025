import re
import subprocess

from rich.console import Console

console = Console()


def strip_markdown_fences(text: str) -> str:
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if m:
        return m.group(1).strip()
    return text.strip()


def run_shell(cmd: str) -> str:
    with console.status(f"⏳ [bold blue]Running shell command:[/] {cmd}", spinner="dots"):
        proc = subprocess.Popen(
            cmd, shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        out, _ = proc.communicate()
    text = out.decode("utf-8", errors="ignore")
    console.log(f"🔹 [bold blue]Shell output:[/]\n{text}")
    return text


def save_to_file(text: str, filename: str):
    console.log(f"💾 Saving to file: [bold green]{filename}[/]")
    with open(filename, "w") as f:
        f.write(text.strip() + "\n")
