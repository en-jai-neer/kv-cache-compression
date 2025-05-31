# Memory Reduction for LLM Inference via KV-Cache Compression

This project implements a novel Key-Value (KV) cache compression strategy for Large Language Models (LLMs), designed to significantly reduce memory demands during long-context inference. As LLMs demonstrate exceptional capabilities in tasks requiring extended context, their memory requirements, particularly for the KV cache, become a critical bottleneck, often exceeding the capacity of current GPUs. This work offers a solution to enable efficient LLM inference in memory-constrained environments without substantially sacrificing generation quality.

## The Challenge: KV Cache Bottleneck in Long-Context LLMs

The KV cache is essential for efficient autoregressive decoding in LLMs, storing past key and value states to calculate attention scores. However, its size scales linearly with context length and batch size, leading to massive memory consumption for long sequences. For instance, a 175 billion parameter model can require over 1,200 GB of GPU memory for a batch size of 64 and a sequence length of 4,096 tokens, far surpassing typical GPU capacities. This limitation hinders the deployment of powerful LLMs for long-context tasks, especially on mobile devices or other memory-restricted settings.

## Our Approach: Merging-Focused, Two-Dimensional Compression

Existing KV cache compression techniques often rely on eviction (discarding tokens) or quantization, which can compromise global context retention or computational efficiency. This project implements a novel approach that emphasizes **merging** over eviction. It features a dual strategy:

1.  **Coarse-Grained Sequence Length Compression**: This strategy progressively merges older tokens using pooling techniques once a user-defined threshold is met. The oldest tokens are compressed more aggressively, while recent tokens are preserved with higher fidelity to maintain crucial information.
2.  **Pyramidal Layer Compression**: Leveraging the observation that lower transformer layers attend to global context and higher layers focus on localized information, this method progressively decreases the compression threshold (and thus the cache budget) across layers. This forms a pyramid-like structure for the KV cache, retaining critical global context in lower layers and focusing on local information in higher layers.

This two-dimensional merging strategy effectively reduces the memory footprint of the KV cache.

## Key Features

* **Memory Efficiency**: Significantly reduces KV cache memory, enabling long-context inference on memory-constrained hardware.
* **Merging-based Compression**: Prioritizes merging tokens over eviction to better preserve global context.
* **Pyramidal Layer Compression**: Adapts cache size per layer, allocating more budget to lower layers for global context and less to higher layers for local context.
* **Sequence Length Compression**: Dynamically merges older tokens along the sequence length using various pooling strategies.
    * Supports pooling types: **mean**, **max**, and **best**. (Our work finds 'Best' pooling often optimal for smaller budgets, and 'Mean' for larger ones).
* **Context Preservation**: Designed to retain critical information by preserving recent tokens and "attention sink" tokens (e.g., initial tokens) without compression.
* **Flexible Configuration**: Allows control over initial cache budget ($L_0$), pyramid compression ratio ($\beta$), number of sink tokens, and pooling strategy.
* **Compatibility**: Built to work with Hugging Face transformer models.

## Methodology Highlights

The compression approach is motivated by several observations:
* Attention sinks (like start tokens) are crucial and are not compressed.
* Attention patterns become more localized in higher transformer layers, justifying progressive compression across layers.
* Beyond initial and recent tokens, there are few attention sinks, making sequence-length compression viable for older tokens.

The **Pyramidal Layer Compression** defines the cache budget for layer $k$ as $L_k = L_0[1 + \frac{k}{m}(\frac{1}{\beta} - 1)]$, where $L_0$ is the initial budget for layer 0, $m$ is the total number of layers, and $\beta$ is the compression ratio.

The **Sequence Length Compression** groups key-value pairs into windows and merges them using strategies like mean, max, or a "best" token selection. Least recent tokens are compressed as new tokens are generated.

## Table of Contents
- [Key-Value Cache Compression for Transformers](#key-value-cache-compression-for-transformers)
  - [Documentation](#documentation)
  - [Requirements](#requirements)
  - [Installation](#installation)
  - [Environment Variables](#environment-variables)
  - [Usage](#usage)
  - [Arguments](#arguments)
  - [Example Usage](#example-usage)
  - [Project Structure](#project-structure)
  - [File Descriptions](#file-descriptions)

## Documentation
- Detailed Report - **[Memory Reduction for LLM Inference via KV-Cache Compression](Memory_Reduction_for_LLM_Inference_via_KV_Cache_Compression.pdf)**

## Requirements
- Python 3.7+
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- torch

Detailed requirements can be found in the `requirements.txt` file.

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/en-jai-neer/kv-cache-compression.git
   cd lm-compression
   ```
2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Environment Variables
Set your Hugging Face API token in your environment to enable access to models:
- Add the token to your shell configuration file (e.g., `.bashrc` or `.zshrc`):
  ```bash
  export HF_API_KEY="your_huggingface_api_token"
  ```

## Usage
Run the script with the desired configuration for model compression and inference:
```bash
python main.py --model_name <model_name> --decoding_strategy <strategy> --initial_local_window <size> ...
```

## Arguments
The following arguments can be passed to the script:

| Argument               | Type    | Default                          | Description                                                                                       |
|------------------------|---------|----------------------------------|---------------------------------------------------------------------------------------------------|
| `--model_name`         | `str`   | `meta-llama/Llama-3-8B-Instruct` | Hugging Face model ID to be used.                                                                 |
| `--decoding_strategy`  | `str`   | `greedy`                         | Decoding strategy for inference. Options: `greedy`, `beam_search`, etc.                           |
| `--initial_local_window`| `int`  | `512`                            | Initial window size of the key-value cache for the first transformer layer.                       |
| `--steepness_coefficient`| `float`| `1.0`                            | Controls the steepness of the local window size decrease across layers.                           |
| `--sink_tokens`        | `int`   | `4`                              | Number of tokens to retain without compression in each layer.                                     |
| `--skip_prefill_compression` | `flag` | -                                | Skips cache compression during the prefill stage if set.                                          |
| `--seq_pooling_type`   | `str`   | `mean`                           | Pooling type for sequence compression. Options: `mean`, `max`, `best`.                            |
| `--compress_context`   | `flag`  | -                                | Enables context compression.                                                                      |
| `--device`             | `str`   | `cpu`                            | Device to be used for inference. Options: `cpu`, `cuda`.                                         |
| `--max_length`         | `int`   | `128`                            | Maximum output sequence length during generation.                                                 |
| `--batch_size`         | `int`   | `4`                              | Batch size for inference.                                                                         |
| `--dataset_split`      | `str`   | `test`                           | Dataset split to be used. Options: `train`, `validation`, `test`.                                 |

## Example Demo
This example demonstrates how to run this compression technique on a user-defined prompt.

```bash
python3 main.py --model_name "meta-llama/Meta-Llama-3-8B-Instruct" \
    --decoding_strategy "greedy" \
    --initial_local_window 128 \
    --steepness_coefficient 1 \
    --sink_tokens 4 \
    --seq_pooling_type "mean" \
    --batch_size 1 \
    --dataset_split "test" \
    --mode "user" \
    --prompt "What is Machine Learning?"
```

## Example Usage
To run the project with a Llama-3-8B-Instruct model, a greedy decoding strategy, and a specific configuration for local window compression:

```bash
python main.py \
    --model_name "meta-llama/Llama-3-8B-Instruct" \
    --decoding_strategy "greedy" \
    --initial_local_window 512 \
    --steepness_coefficient 1.5 \
    --sink_tokens 4 \
    --skip_prefill_compression \
    --seq_pooling_type "mean" \
    --compress_context \
    --device "cuda" \
    --max_length 64 \
    --batch_size 8 \
    --dataset_split "test"
```

After execution, results will be stored in `results` folder.

## Project Structure
```
.
├── cache_dir/                      # Directory for cached data (e.g., model checkpoints) - created during execution.
├── results/                        # Directory for storing results of the experiments (created during execution).
├── src/                            # Source code directory
│   ├── compression/                # Contains code for context and sequence compression.
│   │   ├── context_compress.py     # Contains logic for compressing prompt context.
│   │   └── sequence_kv_compress.py # Contains logic for compressing key-value cache along sequence length for different layers.
│   ├── data/                       # Contains utilities for dataset loading and processing.
│   │   ├── dataset_loader.py       # Contains functions for loading and preparing datasets for model inference.
│   │   └── prompt.py               # Manages prompts and input sequences for model inference.
│   ├── enum/                       # Enums for defining constants such as decoding strategies.
│   │   ├── decoding_strategy.py    # Defines different decoding strategies.
│   │   └── sequence_compression.py # Defines various compression types for KV cache compression.
│   ├── metrics/                    # Contains modules for evaluation metrics.
│   │   ├── longbench_scorer.py     # Longbench performance score generator.
│   │   └── metrics.py              # General metrics for model evaluation used by longbench.
│   ├── model/                      # Defines model utilities.
│   │   ├── huggingface_model.py    # Sets up and configures Hugging Face transformer models.
│   │   └── language_model.py       # Decoding logic with cache-compression support.
│   ├── util/                       # Utilities for execution and result handling.
│   │   ├── execute.py              # Core logic for executing inferences on datasets.
│   │   └── results_io.py           # Handles input/output operations for experiment results.
├── const.py                        # Defines constants used across the project.
├── main.py                         # Main script for configuring and running the project.
└── requirements.txt                # Lists project dependencies.
```

## File Descriptions

### Main Scripts
- **main.py**: The main entry point of the project, which parses command-line arguments and executes the KV cache compression with the specified parameters.

### Modules
- **compression/context_compress.py**: Contains functions for compressing the context information during inference, focusing on reducing the size of the input context.
- **compression/sequence_kv_compress.py**: Implements key-value (KV) cache compression methods, which reduce the cache size while retaining essential information across transformer layers.

- **data/dataset_loader.py**: Loads datasets for the project and prepares them for model inference.
- **data/prompt.py**: Manages prompts and input sequences, which can be passed to the model for inference and evaluation.

- **enum/decoding_strategy.py**: Defines different decoding strategies (e.g., greedy, beam search) as constants or enumerations.
- **enum/sequence_compression.py**: Defines various compression types (e.g., mean, max, best) for use during KV cache compression.

- **metrics/longbench_scorer.py**: Contains custom scoring functions, potentially aligned with the LongBench benchmark, to evaluate model performance over long sequences.
- **metrics/metrics.py**: Provides general metrics used for model evaluation, such as accuracy, F1, or BLEU scores.

- **model/huggingface_model.py**: Sets up and configures Hugging Face transformer models, providing functions to load and initialize models.
- **model/language_model.py**: Implements high-level functions for interacting with language models, making it easier to integrate Hugging Face models with the rest of the project.

- **util/execute.py**: Contains the core logic for executing the main compression task, using the parameters specified by the user.
- **util/results_io.py**: Handles input/output operations for experiment results, such as saving and loading compressed output or evaluation metrics.

### Other Files
- **const.py**: Contains global constants used throughout the project, such as default configuration values or constants used for logging.
- **requirements.txt**: Lists the necessary dependencies for running the project, including libraries such as `torch` and `transformers`.

