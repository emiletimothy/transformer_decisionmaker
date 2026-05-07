"""
Run MWU evaluation via DeepSeek API.

Usage:
  python api_run.py <split> <idx> --model ds|ds_think [--prompt <name>] [--n <runs>] [--plot-weights]

Examples:
  python api_run.py train 0 --model ds
  python api_run.py train 0 --model ds_think
  python api_run.py train 0 --model ds --prompt explicit_update
  python api_run.py train 0 --model ds --n 3   # run 3 times

Models:
  ds        DeepSeek V3 (deepseek-chat)
  ds_think  DeepSeek R1 (deepseek-reasoner, saves reasoning steps)

Available prompts: any .txt file in the prompts/ folder (without extension).
"""

import sys
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))
from mw_lib import load_data, get_case, build_prompt, parse_model_response, DEFAULT_PROMPT_NAME
from run import cmd_evaluate

DATASET_PATH = "mw_dataset.json"

MODEL_MAP = {
    "ds":       "deepseek-chat",
    "ds_think": "deepseek-reasoner",
}

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)


def call_deepseek(prompt, model_key):
    model = MODEL_MAP[model_key]
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    msg = response.choices[0].message
    content = msg.content

    reasoning = None
    if model_key == "ds_think" and hasattr(msg, "reasoning_content"):
        reasoning = msg.reasoning_content

    return content, reasoning


def run_once(split, idx, model_key, prompt_name=DEFAULT_PROMPT_NAME, plot_weights=False):
    data = load_data(DATASET_PATH)
    case = get_case(data, split, idx)
    prompt = build_prompt(case, prompt_name=prompt_name)

    print(f"Calling DeepSeek {model_key.upper()} [{prompt_name}] for {split}_{idx:03d}...")
    content, reasoning = call_deepseek(prompt, model_key)

    try:
        model_output = parse_model_response(content)
    except Exception as e:
        print(f"ERROR: Could not parse model response as JSON.\n  {e}")
        print("--- Raw response ---")
        print(content)
        return None

    # evaluate and save to results/api/train_XXX/<model_key>_<prompt_name>/
    model_name = f"{model_key}_{prompt_name}"
    run_dir, result = cmd_evaluate(split, idx, mode="api", model_output=model_output, model_name=model_name, plot_weights=plot_weights)

    # patch meta with model and prompt info
    result["meta"]["model"] = MODEL_MAP[model_key]
    result["meta"]["model_key"] = model_key
    result["meta"]["prompt"] = prompt_name

    # save reasoning for R1
    if reasoning:
        (run_dir / "reasoning.txt").write_text(reasoning)
        print(f"Reasoning saved to: {run_dir}/reasoning.txt")
        result["reasoning_length"] = len(reasoning)

    # save raw response text
    (run_dir / "raw_response.txt").write_text(content)

    # update result.json with model info
    (run_dir / "result.json").write_text(json.dumps(result, indent=2))

    print(f"Model: {MODEL_MAP[model_key]}")
    return run_dir


def main():
    args = sys.argv[1:]
    if len(args) < 4 or "--model" not in args:
        print(__doc__)
        sys.exit(1)

    split       = args[0]
    idx         = int(args[1])
    model_key   = args[args.index("--model") + 1]
    prompt_name  = args[args.index("--prompt") + 1] if "--prompt" in args else DEFAULT_PROMPT_NAME
    n_runs       = int(args[args.index("--n") + 1]) if "--n" in args else 1
    plot_weights = "--plot-weights" in args

    if model_key not in MODEL_MAP:
        print(f"ERROR: --model must be ds or ds_think, got '{model_key}'")
        sys.exit(1)

    for i in range(n_runs):
        if n_runs > 1:
            print(f"\n=== Run {i+1}/{n_runs} ===")
        run_once(split, idx, model_key, prompt_name, plot_weights=plot_weights)


if __name__ == "__main__":
    main()
