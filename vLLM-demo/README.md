# Run [vLLM](https://github.com/vllm-project/vllm) on Lawrencium Cluster
## Run vLLM on a GPU node on [Lawrencium Cluster](https://it.lbl.gov/service/scienceit/high-performance-computing/)
### Request an [Interactive Jupyter Server](https://it.lbl.gov/resource/hpc/for-users/hpc-documentation/open-ondemand/jupyter-server/) on ES1 GPU partition from [Lawrencium Open OnDemand](https://lrc-ondemand.lbl.gov)
* Connect to Jupyter and open a terminal

# jgi-vllm
### Run JGI vLLM  on Lawrencium
```bash
# To generate sequences on NERSC/LRC, we need a new conda env:

conda create -n vllm python==3.11
conda activate vllm
python -m pip install vllm
```
### Then run this code to test:
Replace **`/path/to/your/model/dir`** with your model.

```python
import os
import transformers
import torch
from vllm import LLM, SamplingParams
from torch.utils.data import Dataset

# set it to prevent the warning message when using the model
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

# Change this to the directory of the model
# LRC:
model_dir = "/path/to/your/model/dir"

"""
Parameters:
- model_dir: str
    The directory of the model
- prompts: List[str]   
    A list of prompts to generate from. If not provided, the model will generate from scratch
- num_generation_from_each_prompt: int
    The number of sequences to generate from each prompt
- temperature: float
    The temperature of the sampling
- min_length: int
    The minimum length of the generated sequence (in tokens)
- max_length: int
    The maximum length of the generated sequence (in tokens)
- top_k: int   
    The top_k parameter for sampling
- presence_penalty: float
    Penalizes new tokens based on whether they appear in the generated text so far. 
    Values > 0 encourage the model to use new tokens, while values < 0 encourage the model to repeat tokens.
- frequency_penalty: float
    Penalizes new tokens based on their frequency in the generated text so far. 
    Values > 0 encourage the model to use new tokens, while values < 0 encourage the model to repeat tokens.
- repetition_penalty: float
    Penalizes new tokens based on whether they appear in the prompt and the generated text so far. 
    Values > 1 encourage the model to use new tokens, while values < 1 encourage the model to repeat tokens.
"""
def generate_sequences(
    model_dir, 
    prompts=[""],
    num_generation_from_each_prompt=100,
    temperature=0.7,
    min_length=128,
    max_length=1024, 
    top_k=50,     
    presence_penalty=0.0,
    frequency_penalty=0.0,
    repetition_penalty=1.0,
):
    llm = LLM(
        model=model_dir,
        tokenizer=model_dir,
        tokenizer_mode="slow",
        trust_remote_code=True,
        seed=0,
        dtype=torch.bfloat16
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir)

    sampling_params = SamplingParams(
        n=num_generation_from_each_prompt,
        temperature=temperature, 
        top_k=top_k,
        stop_token_ids=[2],
        max_tokens=max_length,
        min_tokens=min_length,
        detokenize=False,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        repetition_penalty=repetition_penalty,
    )

    all_outputs = llm.generate(
        prompts=prompts, 
        sampling_params=sampling_params,
    )


    generated_sequences = []
    for outputs in all_outputs:
        for output in outputs.outputs:
            text = tokenizer.decode(output.token_ids, skip_special_tokens=True).replace(" ", "").replace("\n", "")
            generated_sequences.append(text)

    return generated_sequences

"""
Example usage:
"""

# Generate from scratch
print("\n# Generate from scratch\n")
generated_sequences = generate_sequences(model_dir)

# Generate from existing dna sequences
print("\n# Generate from existing dna sequences\n") 
dna_sequences = ["ATAGCATGATGTACG", "GCTCAGTGCTAGCAA"] # save the sequences in a list
generate_sequences(model_dir, prompts=dna_sequences)
```

### On Lawrencium: 
- The Lawrencium A40 GPU, with its 46 GiB of memory, can run either **`generated_sequences = generate_sequences(model_dir)`** or **`generate_sequences(model_dir, prompts=dna_sequences)`**, but attempting to run both simultaneously results in a CUDA out of memory error.

#### Output from `generate_sequences(model_dir, prompts=dna_sequences)`:
```
# # Generate from existing dna sequences 

INFO 06-24 14:24:11 llm_engine.py:161] Initializing an LLM engine (v0.5.0.post1) with config: model='/path/to/your/model/dir', speculative_config=None, tokenizer='/path/to/your/model/dir', skip_tokenizer_init=False, tokenizer_mode=slow, revision=None, rope_scaling=None, rope_theta=None, tokenizer_revision=None, trust_remote_code=True, dtype=torch.bfloat16, max_seq_len=32768, download_dir=None, load_format=LoadFormat.AUTO, tensor_parallel_size=1, disable_custom_all_reduce=False, quantization=None, enforce_eager=False, kv_cache_dtype=auto, quantization_param_path=None, device_config=cuda, decoding_config=DecodingConfig(guided_decoding_backend='outlines'), seed=0, served_model_name=/path/to/your/model/dir)
INFO 06-24 14:24:15 model_runner.py:160] Loading model weights took 7.9378 GB
INFO 06-24 14:24:19 gpu_executor.py:83] # GPU blocks: 19000, # CPU blocks: 2730
INFO 06-24 14:24:20 model_runner.py:889] Capturing the model for CUDA graphs. This may lead to unexpected consequences if the model is not static. To run the model in eager mode, set 'enforce_eager=True' or use '--enforce-eager' in the CLI.
INFO 06-24 14:24:20 model_runner.py:893] CUDA graphs can take additional 1~3 GiB memory per GPU. If you are running out of memory, consider decreasing `gpu_memory_utilization` or enforcing eager mode. You can also reduce the `max_num_seqs` as needed to decrease memory usage.
INFO 06-24 14:24:28 model_runner.py:965] Graph capturing finished in 8 secs.
Processed prompts: 100%|█| 2/2 [00:24<00:00, 12.40s/it, est. speed input: 0.48 toks/s, ou

```
