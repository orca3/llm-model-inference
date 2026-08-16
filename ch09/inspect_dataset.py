#!/usr/bin/env python3
"""Inspect the benchmark datasets used by `vllm bench serve`.

This helper loads a dataset with the very same code path that
`vllm bench serve` uses (``vllm.benchmarks.datasets``), then reports the
prompt/output length distributions, prints a few sample prompts and draws a
simple ASCII histogram. It is meant to answer "what traffic am I actually
sending to the server?" before a benchmark run.

Copy this file into the root of your local `vllm` repository (see README.md)
and run, for example:

    python3 inspect_dataset.py \
        --dataset-name sharegpt \
        --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
        --model Qwen/Qwen3-14B \
        --num-prompts 100 \
        --save-samples

    python3 inspect_dataset.py \
        --dataset-name prefix_repetition \
        --model Qwen/Qwen3-14B \
        --num-prompts 50 \
        --prefix-repetition-prefix-len 256 \
        --prefix-repetition-suffix-len 256 \
        --prefix-repetition-num-prefixes 5 \
        --prefix-repetition-output-len 128 \
        --save-samples
"""

import argparse
import json

import numpy as np

from vllm.benchmarks.datasets import (
    PrefixRepetitionRandomDataset,
    RandomDataset,
    ShareGPTDataset,
)
try:
    # vLLM >= 0.12: tokenizer helpers live in their own package.
    from vllm.tokenizers import get_tokenizer
except ImportError:
    from vllm.transformers_utils.tokenizer import get_tokenizer

# Number of buckets in the ASCII histogram.
NUM_HISTOGRAM_BINS = 9
# Prompts longer than this are truncated when printed.
MAX_PROMPT_CHARS = 200
# How many sample prompts to print.
NUM_SAMPLES_TO_SHOW = 5


def build_dataset(args, tokenizer):
    """Sample `args.num_prompts` requests from the requested dataset."""
    if args.dataset_name == "sharegpt":
        if not args.dataset_path:
            raise ValueError("--dataset-path is required for the sharegpt dataset")
        dataset = ShareGPTDataset(
            dataset_path=args.dataset_path,
            random_seed=args.seed,
        )
        return dataset.sample(
            tokenizer=tokenizer,
            num_requests=args.num_prompts,
            output_len=args.sharegpt_output_len,
            request_id_prefix=args.request_id_prefix,
        )

    if args.dataset_name == "prefix_repetition":
        dataset = PrefixRepetitionRandomDataset(random_seed=args.seed)
        return dataset.sample(
            tokenizer=tokenizer,
            num_requests=args.num_prompts,
            prefix_len=args.prefix_repetition_prefix_len,
            suffix_len=args.prefix_repetition_suffix_len,
            num_prefixes=args.prefix_repetition_num_prefixes,
            output_len=args.prefix_repetition_output_len,
            request_id_prefix=args.request_id_prefix,
        )

    if args.dataset_name == "random":
        dataset = RandomDataset(random_seed=args.seed)
        return dataset.sample(
            tokenizer=tokenizer,
            num_requests=args.num_prompts,
            prefix_len=args.random_prefix_len,
            input_len=args.random_input_len,
            output_len=args.random_output_len,
            range_ratio=args.random_range_ratio,
            request_id_prefix=args.request_id_prefix,
        )

    raise ValueError(f"Unknown dataset name: {args.dataset_name}")


def print_distribution(name, values):
    """Print min/max/mean/median/std for one length distribution."""
    print(f"\n=== {name} Length Distribution ===")
    print(f"Min {name.lower()} length: {int(np.min(values))}")
    print(f"Max {name.lower()} length: {int(np.max(values))}")
    print(f"Mean {name.lower()} length: {np.mean(values):.2f}")
    print(f"Median {name.lower()} length: {np.median(values):.2f}")
    print(f"Std {name.lower()} length: {np.std(values):.2f}")


def print_samples(requests, num_samples):
    """Print the first few requests with their prompt truncated."""
    print("\n=== Sample Prompts ===")
    for i, request in enumerate(requests[:num_samples]):
        prompt = request.prompt
        if not isinstance(prompt, str):
            prompt = str(prompt)
        if len(prompt) > MAX_PROMPT_CHARS:
            prompt = prompt[:MAX_PROMPT_CHARS] + "..."
        print(f"\n--- Sample {i + 1} ---")
        print(f"Prompt length: {request.prompt_len}")
        print(f"Output length: {request.expected_output_len}")
        print(f"Request ID: {getattr(request, 'request_id', None)}")
        print(f"Prompt: {prompt}")


def print_histogram(prompt_lens, num_bins=NUM_HISTOGRAM_BINS):
    """Draw an ASCII histogram of the prompt length distribution."""
    print("\n=== Prompt Length Histogram ===")
    min_len = int(np.min(prompt_lens))
    max_len = int(np.max(prompt_lens))

    if min_len == max_len:
        # Every prompt has the same length (the usual case for the synthetic
        # datasets): a histogram of empty bins would be noise, so print one row.
        print(f"{min_len:4d} tokens: {'*' * len(prompt_lens)}")
        return

    bin_width = (max_len - min_len) / num_bins
    counts = [0] * num_bins
    for length in prompt_lens:
        index = min(int((length - min_len) / bin_width), num_bins - 1)
        counts[index] += 1

    for i, count in enumerate(counts):
        low = int(min_len + i * bin_width)
        high = int(min_len + (i + 1) * bin_width)
        print(f"{low:4d}-{high:4d} tokens: {'*' * count}")


def save_samples(requests, filename):
    """Dump the sampled requests to JSON for offline inspection."""
    samples = [
        {
            "request_id": getattr(request, "request_id", None),
            "prompt": request.prompt,
            "prompt_len": request.prompt_len,
            "expected_output_len": request.expected_output_len,
            "has_multimodal": bool(getattr(request, "multi_modal_data", None)),
        }
        for request in requests
    ]
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(samples)} samples to {filename}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect a vLLM benchmark dataset before running a benchmark."
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="sharegpt",
        choices=["sharegpt", "prefix_repetition", "random"],
        help="Which benchmark dataset to load.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to the dataset file (required for sharegpt).",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name or path used to load the tokenizer.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Tokenizer name or path (defaults to --model).",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when loading the tokenizer.",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=100,
        help="Number of prompts to sample from the dataset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used when sampling the dataset.",
    )
    parser.add_argument(
        "--request-id-prefix",
        type=str,
        default="",
        help="Prefix prepended to generated request IDs.",
    )
    parser.add_argument(
        "--num-samples-to-show",
        type=int,
        default=NUM_SAMPLES_TO_SHOW,
        help="How many sample prompts to print.",
    )
    parser.add_argument(
        "--save-samples",
        action="store_true",
        help="Save the sampled requests to a JSON file.",
    )
    parser.add_argument(
        "--samples-filename",
        type=str,
        default=None,
        help="Output file for --save-samples "
        "(defaults to <dataset-name>_samples.json).",
    )

    # ShareGPT specific.
    parser.add_argument(
        "--sharegpt-output-len",
        type=int,
        default=None,
        help="Override the output length of every ShareGPT request.",
    )

    # Prefix repetition specific.
    parser.add_argument("--prefix-repetition-prefix-len", type=int, default=256)
    parser.add_argument("--prefix-repetition-suffix-len", type=int, default=256)
    parser.add_argument("--prefix-repetition-num-prefixes", type=int, default=10)
    parser.add_argument("--prefix-repetition-output-len", type=int, default=128)

    # Random dataset specific.
    parser.add_argument("--random-input-len", type=int, default=1024)
    parser.add_argument("--random-output-len", type=int, default=128)
    parser.add_argument("--random-prefix-len", type=int, default=0)
    parser.add_argument("--random-range-ratio", type=float, default=0.0)

    return parser.parse_args()


def main():
    args = parse_args()

    tokenizer = get_tokenizer(
        args.tokenizer or args.model,
        trust_remote_code=args.trust_remote_code,
    )

    print(f"Loading dataset: {args.dataset_name}")
    requests = build_dataset(args, tokenizer)

    print("\n=== Dataset Overview ===")
    print(f"Total samples: {len(requests)}")
    if not requests:
        return

    prompt_lens = [r.prompt_len for r in requests]
    output_lens = [r.expected_output_len for r in requests]

    print_distribution("Prompt", prompt_lens)
    print_distribution("Output", output_lens)
    print_samples(requests, args.num_samples_to_show)
    print_histogram(prompt_lens)

    if args.save_samples:
        filename = args.samples_filename or f"{args.dataset_name}_samples.json"
        save_samples(requests, filename)


if __name__ == "__main__":
    main()
