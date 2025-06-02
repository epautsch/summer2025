from pipelines.dev_pipeline import DevPipeline
import os
import torch


if __name__ == "__main__":
    if torch.cuda.is_available():
        print("CUDA available. Assume compute node with no network connection. Enabling HF offline mode.")
        os.environ["HF_HUB_OFFLINE"] = "1"
    else:
        print("CUDA NOT available. Assume login node with network connection. Disabling HF offline mode.")
        os.environ["HF_HUB_OFFLINE"] = "0"
    
    #os.environ["HF_HOME"] = "/grand/EVITA/erik/transformers_cache"

    pipeline = DevPipeline()
    task = "Write a function that calculates the Fibonacci sequence."
    result = pipeline.run(task)
    print("\nFinal result:\n", result)

