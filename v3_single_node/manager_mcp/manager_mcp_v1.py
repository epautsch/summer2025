import asyncio
import json
import logging
import os
import re
import shutil
from contextlib import AsyncExitStack
from typing import Any

import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from rich.console import Console
from rich.traceback import install
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.status import Status


# Init rich console
console = Console()
install()

# logging setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# model and preocessor setup
torch_dtype = "auto"
model_id = "google/gemma-3-27b-it"
model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id,
    attn_implementation="eager",
    device_map="auto",
    torch_dtype=torch.bfloat16,
).eval()
processor = AutoProcessor.from_pretrained(model_id)

class Configuration:
    # load up server config from JSON file
    @staticmethod
    def load_config(file_path: str) -> dict[str, Any]:
        with open(file_path, "r") as f:
            return json.load(f)

class Server:
    # Manages an mcp server process and sessions
    def __init__(self, name: str, config: dict[str, Any]):
        self.name = name
        self.config = config
        self.session: ClientSession | None = None
        self.exit_stack: AsyncExitStack = AsyncExitStack()

    async def initialize(self) -> None:
        command = (
            shutil.which(self.config["command"]) or self.config["command"]
        )
        server_params = StdioServerParameters(
            command=command,
            args=self.config.get("args", []),
            env={**os.environ, **self.config.get("env", {})},
        )
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        read, write = stdio_transport
        session = await self.exit_stack.enter_async_context(
            ClientSession(read, write)
        )
        await session.initialize()
        self.session = session
        console.log(f"[green]Initialized server {self.name}[/]")

    async def list_tools(self) -> list[Any]:
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")
        resp = await self.session.list_tools()
        tools = []
        for kind, entries in resp:
            if kind == "tools":
                tools.extend(
                    Tool(tool.name, tool.description, tool.inputSchema)
                    for tool in entries
                )
        return tools

    async def execute_tool(self, tool_name: str, args: dict[str, Any]) -> Any:
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")
        console.log(f"[cyan]Executing tool[/] {tool_name} with args {args}")
        return await self.session.call_tool(tool_name, args)

    async def cleanup(self) -> None:
        await self.exit_stack.aclose()
        self.session = None
        console.log(f"[red]Cleaned up server {self.name}[/]")

class Tool:
    # metadata wrapper for server-exposed tool
    def __init__(self, name: str, description: str, input_schema: dict[str, Any]):
        self.name = name
        self.description = description
        self.input_schema = input_schema

    def format_for_llm(self) -> str:
        parts = [f"Tool: {self.name}", f"Description: {self.description}"]
        props = self.input_schema.get("properties", {})
        required = set(self.input_schema.get("required", []))
        if props:
            parts.append("Arguments:")
            for pname, info in props.items():
                desc = info.get("description", "No description")
                req = " (required)" if pname in required else ""
                parts.append(f"- {pname}: {desc}{req}")
        return "\n".join(parts)

class LocalLLMClient:
    # handles chat completions using the local model
    def __init__(self, model, processor, max_new_tokens: int = 2048):
        self.model = model
        self.processor = processor
        self.max_new_tokens = max_new_tokens

    def get_response(self, messages: list[dict[str, str]]) -> str:
        formatted = []
        for m in messages:
            formatted.append({
                "role": m["role"],
                "content": [
                    {"type": "text", "text": m["content"]}
                ]
            })
        
        with console.status("Generating response...", spinner="dots"):
            tokenized = self.processor.apply_chat_template(
                formatted,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device, dtype=torch.bfloat16)

            input_ids = tokenized["input_ids"]
            input_len = input_ids.shape[-1]
            with torch.inference_mode():
                out = self.model.generate(
                    **tokenized,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )
        gen = out[0][input_len:]
        text = self.processor.decode(gen, skip_special_tokens=True)
        text = re.sub(r"^json\s*", "", text, flags=re.IGNORECASE).lstrip()
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()

class ChatSession:
    # orchestrates the repl loop, llm, and tool calls
    def __init__(self, servers: list[Server], llm: LocalLLMClient):
        self.servers = servers
        self.llm = llm

    async def cleanup_servers(self) -> None:
        for server in reversed(self.servers):
            await server.cleanup()

    async def process_llm_response(self, llm_response: str) -> str:
        try:
            tool_call = json.loads(llm_response)
            if "tool" in tool_call:
                args = tool_call.get("arguments", {})
                for server in self.servers:
                    tools = await server.list_tools()
                    if any(t.name == tool_call["tool"] for t in tools):
                        result = await server.execute_tool(tool_call["tool"], args)
                        return json.dumps(
                            result,
                            default=lambda o: o.__dict__ if hasattr(o, '__dict__') else str(o),
                        )
            return llm_response
        except json.JSONDecodeError:
            return llm_response

    async def start(self) -> None:
        for srv in self.servers:
            await srv.initialize()
        all_tools = []
        for srv in self.servers:
            all_tools.extend(await srv.list_tools())
        tools_desc = "\n\n".join(t.format_for_llm() for t in all_tools)

        system_msg = (
            "You are a helpful assistant with access to these tools:\n\n"
            f"{tools_desc}\n"
            "Choose the appropriate tool based on the user's question. "
            "If no tool is needed, reply directly.\n\n"
            "IMPORTANT: When you need to use a tool, you must ONLY respond with "
            "the exact JSON object format below, nothing else:\n"
            "{\n"
            '   "tool": "tool-name",\n'
            '   "arguments": {\n'
            '       "argument-name": "value"\n'
            "   }\n"
            "}\n\n"
            "After receiving a tool's response:\n"
            "1. Transform the raw data into a natural, conversational response\n"
            "2. Keep responses concise but informative\n"
            "3. Focus on the most relevant information\n"
            "4. Use appropriate context from the user's question\n"
            "5. Avoid simply repeating the raw data\n\n"
            "Please use only the tools that are explicitly defined above."
        )
        messages = [{"role": "system", "content": system_msg}]

        while True:
            user_input = console.input("[bold cyan]> [/]").strip()
            if user_input.lower() in ("quit", "exit"):
                console.print("[bold red]Goodbye![/]")
                break

            messages.append({"role": "user", "content": user_input})
            resp = self.llm.get_response(messages)
            console.print(Panel(resp, title="Assistant"))

            tool_result = await self.process_llm_response(resp)
            if tool_result != resp:
                messages.append({"role": "assistant", "content": resp})
                messages.append({"role": "user", "content": tool_result})
                final = self.llm.get_response(messages)
                console.print(Panel(final, title="Assistant"))
                messages.append({"role": "assistant", "content": final})
            else:
                messages.append({"role": "assistant", "content": resp})

        await self.cleanup_servers()

async def main() -> None:
    config = Configuration.load_config("servers_config.json")
    servers = [Server(name, cfg) for name, cfg in config.get("mcpServers", {}).items()]
    llm_client = LocalLLMClient(model, processor)
    session = ChatSession(servers, llm_client)
    await session.start()

if __name__ == "__main__":
    asyncio.run(main())

