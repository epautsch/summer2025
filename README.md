# Project Progress Report

**Change Log**
- **2025-06-10**: Initial progress report created.

## 1. Overview

A brief summary of where we stand in building an “agentic” system for HPC software development using large language models (LLMs), particularly Google Gemma 3 (27B). We have prototyped a custom pipeline with “Analyst → Coder → Builder → Tester” agents and started exploring Smolagents for higher-level orchestration.

## 2. Achievements to Date

1. **Custom Agentic Pipeline with Raw Transformers**  
   - Successfully implemented a multi-agent workflow using the HuggingFace Transformers library and the `gemma3-27b-it` checkpoint.  
   - Structured sequential calls:  
     - **Analyst** (generates a high-level software specification)  
     - **Coder** (writes CUDA code from the spec)  
     - **Builder** (creates CMake files)  
     - **Tester** (generates unit tests)  
   - Verified end-to-end on a single Polaris node (4 GPUs) by sharding Gemma 3 across all four devices via `device_map="auto"`.

2. **GPU Memory Considerations**  
   - Discovered that careful tuning of `max_new_tokens` is crucial: each agent’s prompt length plus new tokens must stay within VRAM limits.  
   - Noted that when the **Builder** consumes a long code output from the **Coder**, its own output budget must be smaller to avoid “misaligned address” or OOM errors.

3. **Single-Node Model-Parallel Setup**  
   - Used a custom wrapper to load Gemma 3 on all 4 GPUs (model parallel), enabling the pipeline to run in a single process while pipelining across devices.  
   - Achieved stable runs for the Analyst → Coder portion; Builder/Tester sometimes hit alignment/OOM errors when `max_new_tokens` was too high.

4. **Initial Attempts to Scale to Two Nodes (8 GPUs)**  
   - Tried both **Accelerate** and **torchrun** to shard Gemma 3 across 8 GPUs on 2 Polaris nodes.  
   - Encountered rendezvous/time-out issues: only one node’s processes would join the TCP rendezvous, causing “1/2 clients joined” errors.  
   - Confirmed that none of the alternative LLaMA-based models matched Gemma 3’s quality for our coding tasks.

## 3. Challenges & Pain Points

1. **Token-Length & Alignment Errors**  
   - Longer intermediate outputs (especially from the Coder) push subsequent agents over their GPU memory limit, causing:  
     - `RuntimeError: misaligned address`  
     - OOM on KV-cache allocation  
   - Requires careful slicing of prompt + generation budgets per agent (e.g. Coder’s `max_new_tokens=768` vs. Builder’s `max_new_tokens=512`).

2. **Multi-Node Rendezvous Issues**  
   - Both Accelerate and torchrun setups failed to bring up all 8 ranks reliably: one node never joined the rendezvous.  
   - Spent significant time debugging `MASTER_ADDR`/`MASTER_PORT` environment variables, PBS settings, and NCCL configuration.  
   - Still not fully resolved, so two-node runs remain unstable.

3. **Model Quality & Determinism**  
   - LLaMA-family checkpoints tried (e.g., various LLaMA 2 versions) underperformed Gemma 3 in generating coherent code specs.  
   - Even with Gemma 3, results can be non-deterministic: only ~1 of 6 runs produced a meaningful “final thought” indicator.  
   - Generated software spec by Analyst wasn’t always high-quality—sometimes missing key CUDA/HPC details.

4. **Early Smolagents Exploration**  
   - Began studying Smolagents documentation (guided tour, tutorials, tools).  
   - Wrote prototype Smolagents code (Analyst agent with WebSearchTool) but ran into various CUDA/kernel errors when loading Gemma 3 under Smolagents.  
   - Smolagents examples (RAG, ReAct) inspired ideas, but implementing them precisely led to additional alignment/port/time-out issues.

## 4. Smolagents: Promising Directions & Outstanding Questions

- **Why Smolagents?**  
  - Built-in “agent orchestration” (manager + sub-agents) could eliminate boilerplate loops over `model.generate()`.  
  - Easier to attach tools (WebSearch, file I/O, shell execution) so that agents can “think” and “act” (ReAct style).  
- **Current Roadblocks**  
  1. **CUDA Errors under Smolagents**:  
     - Attempting to load Gemma 3 inside a `TransformersModel` wrapper triggered “misaligned address” or out-of-memory errors.  
     - Sometimes only 1 out of 6 attempts succeeded in producing a final output.  
  2. **Lack of Determinism & Quality**:  
     - Analyst outputs were too generic; not specialized enough for HPC/CUDA.  
     - WebSearchTool usage was hit-or-miss—sometimes the LLM ignored it entirely.  
- **ReAct & Iterative Loops**  
  - Smolagents’ ReAct approach promises “think-action-observe” loops:  
    1. Agent writes a snippet of code (or a `!run_shell` or `!save_file` command).  
    2. The tool executes it and returns an observation.  
    3. Agent “reflects” on that observation, then decides next step.  
  - We must implement secure, sandboxed code execution (Docker, limited privileges) if we allow unverified LLM-generated shell code.  
- **RAG (Retrieval-Augmented Generation)**  
  - Smolagents RAG example suggests hooking up a local knowledge base of HPC docs.  
  - We could pre-index CUDA runtime docs, MPI docs, etc., so the Analyst or Coder can retrieve relevant snippets rather than blindly copying from the Internet.

## 5. Next Steps & Action Items

1. **Stabilize Single-Node Smolagents Prototype**  
   - Focus on **just the Analyst agent** (see earlier prototype).  
   - Confirm that Gemma 3 can load on **all 4 GPUs** via a custom sharding wrapper, then call Smolagents’ `CodeAgent` without kernel errors.  
   - Tune `max_new_tokens` for the Analyst’s spec: 512 is likely sufficient.

2. **Build & Test Custom Tools**  
   - **SaveFileTool**: accept a filename + content, write to disk.  
   - **RunShellTool**: execute `nvcc`, `cmake`, `make`, `ctest`, capture stdout/stderr.  
   - Ensure these tools handle failures gracefully (nonzero exit codes).

3. **Design ReAct Loop for Multi-Agent Pipeline**  
   - Manager agent uses ReAct: “Step 1: call Analyst. Wait for spec. Step 2: call Coder with spec. …”  
   - If Builder’s compilation fails, ReAct loop routes “compilation error” back to Coder to fix code.  
   - Add a simple feedback mechanism: store last error, prepend to next call’s prompt (e.g. “Fix the following compile errors: …”).

4. **Experiment with Single-Node Multi-GPU Sharding**  
   - Write a custom `Gemma3ShardedModel` (as shown previously) that uses `device_map="auto"`.  
   - Launch `python analyst_only.py` (or `gemma_smol_pipeline.py`) with `torchrun --nproc_per_node=4`.  
   - Verify via `nvidia-smi` that all four GPUs show roughly ¼ of Gemma 3’s memory usage.

5. **Progressively Add Coder, Builder, Tester Agents**  
   - Once the Analyst prototype is stable, add the Coder agent (take spec → produce `matmul.cu`, call `SaveFileTool`).  
   - Then Builder agent (take `matmul.cu` → generate `CMakeLists.txt`, call `SaveFileTool` → optionally run `run_shell("cmake . && make -j8")`).  
   - Finally, Tester agent (generate `test_matmul.cpp`, call `SaveFileTool` → `run_shell("ctest --output-on-failure")`).  
   - Keep each agent’s `max_new_tokens` small to avoid memory spikes (e.g. Coder ≤ 1024, Builder ≤ 512, Tester ≤ 1024).

6. **Evaluate Quality & Determinism**  
   - Run multiple Analyst→Coder→Builder→Tester pipelines with the same prompt to gauge variance.  
   - If outputs vary widely, seed the LLM or add stronger constraints in each agent’s description.  
   - Experiment with different decoding strategies (`do_sample=False` vs. `top_k`/`top_p`) to improve consistency.

7. **Investigate RAG for HPC Documentation**  
   - Pre-index local PDF/HTML dumps of CUDA, MPI, and BLAS documentation.  
   - Build a lightweight FAISS index or use HF’s `datasets` + `retrieval` tools.  
   - Integrate a **RetrievalTool** so that Analyst/Coder can fetch specific code examples or API descriptions on demand.

8. **Prepare Conference & Research Integrations**  
   - Look up “Software Engineering Papers at SC24” (the call for submissions at Cyprus). Summarize relevant ones.  
   - Explore how to integrate **MCP (Model-Conditioned Prompting)** into this agentic design.  
   - Consider the **GPU2JPEG (CUDA→SYCL) project** for Aurora—how would an agent approach rewriting CUDA kernels to SYCL?  
   - Read the **ChatVis SC24** paper to see if visualization of agent workflows could improve debugging.

## 6. References & Useful Links

- **Smolagents Documentation**  
  - Guided tour: https://huggingface.co/docs/smolagents/en/guided_tour  
  - Building good agents: https://huggingface.co/docs/smolagents/en/tutorials/building_good_agents  
  - Tools tutorial: https://huggingface.co/docs/smolagents/en/tutorials/tools  

- **Gemma 3 Model**  
  - HF checkpoint: `google/gemma-3-27b-it`  

- **Polaris / ALCF HPC Docs**  
  - Multi-GPU scaling: details on NCCL, libfabric, PMI-based GPU affinity, etc.  

- **Key Next-Step Resources**  
  - “ReAct: Synergize Reasoning and Acting in Language Models” (Smolagents reference for thought/tool loops)  
  - FAISS + In-Context Retrieval tutorials for building RAG pipelines

## 7. Conclusion

We have a working single-node, 4-GPU “Analyst → Coder” pipeline using raw HF Gemma 3. The main bottleneck now is:

- Stabilizing Smolagents’ environment so that Gemma 3 loads reliably in a `CodeAgent`.  
- Handling CUDA memory vs. `max_new_tokens` trade-offs at each agent.  
- Building out the ReAct loop with custom Tools for file I/O and shell execution.

Once the Analyst prototype is rock solid, we will incrementally add Coder, Builder, and Tester agents, then revisit multi-node model-parallel scaling if additional VRAM is needed. Finally, we’ll integrate RAG, MCP, and visualize workflows for our HPC CUDA “code factory” use case.
