import os
from smolagents import CodeAgent, WebSearchTool, TransformersModel


model = TransformersModel(
       #model_id="google/gemma-3-4b-it",
       #model_id="google/gemma-3-12b-it",
       model_id="google/gemma-3-27b-it",
       device_map="auto",
       max_new_tokens=256
)

search_tool = WebSearchTool()

analyst_agent = CodeAgent(
        name="hpc_analyst",
        description=(
            "You are an HPC software analyst. Your job is to take a user's high-level task "
            '(e.g. "Implement matrix multiplication in CUDA") and produce a concise, step-by-step '
            "software specification (no code) that a single coder could implement. "
            "Keep the specification under 200 words and in 10 bullet points or fewer. "
            "If you need to look up CUDA or HPC documentation, call the `run_web_search(query)` tool."
        ),
        tools=[search_tool],
        model=model,
        stream_outputs=False
)

if __name__ == "__main__":
    user_task = "Come up witha software specification a coder can use to implement matrix multiplication in CUDA."

    spec = analyst_agent.run(user_task)

    print("\n===== Analyst's Specification =====\n")
    print(spec)
