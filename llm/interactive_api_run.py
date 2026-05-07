"""
Run MWU evaluation interactively via DeepSeek API.

Usage:
  python interactive_api_run.py <split> --idx <i ...> --model <m ...> --prompt <p ...>
                                [--dataset <path>] [--plot-weights]

  --idx     One or more case indices, or "all" for every case in the split.
  --model   One or more model keys (ds, ds_think).
  --prompt  One or more prompt names (files in prompts/ without .txt).
  --dataset Path to dataset JSON (default: mw_dataset_test.json).

Already-completed (split, idx, model, prompt) combos are skipped automatically.

Examples:
  # single case
  python interactive_api_run.py cases --idx 0 --model ds --prompt interactive_online

  # batch: all 3 cases × 2 prompts
  python interactive_api_run.py cases --idx all --model ds --prompt interactive_online interactive_weather

  # multiple models and prompts
  python interactive_api_run.py cases --idx 0 1 2 --model ds --prompt interactive_online interactive_weather

Models:
  ds            DeepSeek V3 (deepseek-chat)
  ds_think      DeepSeek R1 (deepseek-reasoner)
  gpt4o         GPT-4o
  gpt4o_mini    GPT-4o-mini
  o3_mini       o3-mini
  gemini_flash  Gemini 2.5 Flash
  gemini_pro    Gemini 2.5 Pro

Results are saved under:
  results/<prompt_short>/<model>/<split>_<idx>/run001/
  results/<prompt_short>/<model>/regret.xlsx
"""

import sys
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

from mw_lib import (
    load_data, get_case, load_prompt_template, parse_model_response,
    analyze_model_output, get_ground_truth_outputs,
    analyze_online_output, plot_online_regret, plot_online_loss,
    plot_regret_comparison, plot_weight_comparison, plot_cumulative_loss,
)
try:
    from run import _write_report_notebook
except ImportError:
    _write_report_notebook = None  # archived, not needed for two-turn mode

DATASET_PATH = "mw_dataset.json"
RESULTS_DIR  = Path(os.environ.get("RESULTS_DIR", "exp_results"))
DEFAULT_PROMPT = "interactive_weather"

MODEL_MAP = {
    "ds":            "deepseek-chat",
    "ds_think":      "deepseek-reasoner",
    "gpt4o":         "gpt-4o",
    "gpt4o_mini":    "gpt-4o-mini",
    "gpt54":         "gpt-5.4",
    "gpt54_mini":    "gpt-5.4-mini",
    "gpt54_mini_reason": "gpt-5.4-mini",
    "gpt54_nano":    "gpt-5.4-nano",
    "o3_mini":       "o3-mini",
    "o4_mini":       "o4-mini",
    "gemini25_flash": "gemini-2.5-flash",
    "gemini25_pro":   "gemini-2.5-pro",
    "gemini3_flash":  "gemini-3-flash-preview",
    "gemini3_pro":    "gemini-3-pro-preview",
    "gemini31_pro":   "gemini-3.1-pro-preview",
    "qwen3_14b":          "Qwen3-14B",   # thinking (budget=1024)
    "qwen3_14b_think512": "Qwen3-14B", # thinking (budget=512)
    "qwen3_14b_think1024":"Qwen3-14B", # thinking (budget=1024)
    "qwen3_14b_think2048":"Qwen3-14B", # thinking (budget=2048)
    "qwen3_14b_nothink":  "Qwen3-14B", # non-thinking mode
}

_DEEPSEEK_MODELS = {"ds", "ds_think"}
_OPENAI_MODELS   = {"gpt4o", "gpt4o_mini", "gpt54", "gpt54_mini", "gpt54_mini_reason", "gpt54_nano", "o3_mini", "o4_mini"}
_LOCAL_MODELS    = {"qwen3_14b", "qwen3_14b_think512", "qwen3_14b_think1024",
                    "qwen3_14b_think2048", "qwen3_14b_nothink"}
_REASONING_MODELS = {"o3_mini", "o4_mini", "gpt54_mini_reason"}  # models with reasoning_effort
_GEMINI_MODELS   = {"gemini25_flash", "gemini25_pro", "gemini3_flash", "gemini3_pro", "gemini31_pro"}

_clients = {}

def _get_client(model_key):
    if model_key in _DEEPSEEK_MODELS:
        provider = "deepseek"
    elif model_key in _GEMINI_MODELS:
        provider = "gemini"
    elif model_key in _LOCAL_MODELS:
        provider = "local"
    else:
        provider = "openai"
    if provider not in _clients:
        if provider == "deepseek":
            _clients[provider] = OpenAI(
                api_key=os.environ["DEEPSEEK_API_KEY"],
                base_url="https://api.deepseek.com",
            )
        elif provider == "gemini":
            _clients[provider] = OpenAI(
                api_key=os.environ["GEMINI_API_KEY"],
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            )
        elif provider == "local":
            _clients[provider] = OpenAI(
                api_key="none",
                base_url="http://localhost:8000/v1",
            )
        else:
            _clients[provider] = OpenAI(
                api_key=os.environ["OPENAI_API_KEY"],
            )
    return _clients[provider]


# ── timing log ───────────────────────────────────────────────────────────────

TIMING_LOG = RESULTS_DIR / "timing.log"

def _log_timing(msg):
    from datetime import datetime
    TIMING_LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    with open(TIMING_LOG, "a") as f:
        f.write(line + "\n")


# ── directory helpers ─────────────────────────────────────────────────────────

def _prompt_short(prompt_name):
    """Strip leading 'interactive_' → e.g. 'explicit_update', 'online'."""
    return prompt_name.removeprefix("interactive_")


def _label(model_key, prompt_name):
    """Legacy helper kept for print messages."""
    return f"{model_key}_{_prompt_short(prompt_name)}"


def get_run_dir(split, idx, model_key, prompt_name):
    """
    results/explicit_update/ds/cases_000/run001/
    """
    short  = _prompt_short(prompt_name)
    case_d = RESULTS_DIR / short / model_key / f"{split}_{idx:03d}"
    case_d.mkdir(parents=True, exist_ok=True)

    existing = sorted(case_d.glob("run*/"))
    if not existing:
        return case_d / "run001"
    last = int(existing[-1].name.removeprefix("run"))
    return case_d / f"run{last + 1:03d}"


def has_existing_run(split, idx, model_key, prompt_name):
    """Return True if at least one *completed* run exists (has result.json)."""
    short  = _prompt_short(prompt_name)
    case_d = RESULTS_DIR / short / model_key / f"{split}_{idx:03d}"
    if not case_d.exists():
        return False
    return any(case_d.glob("run*/result.json"))


def _update_regret_table(model_key, prompt_name, split, idx, regret_curve):
    """Append / update one row in the regret Excel table.
    Saved at: results/<prompt_short>/<model_key>/regret.xlsx
    """
    short    = _prompt_short(prompt_name)
    model_d  = RESULTS_DIR / short / model_key
    model_d.mkdir(parents=True, exist_ok=True)
    xlsx     = model_d / "regret.xlsx"
    row_name = f"{split}_{idx:03d}"

    if xlsx.exists():
        df = pd.read_excel(xlsx, index_col=0)
    else:
        df = pd.DataFrame()

    n_steps = len(regret_curve)
    cols = list(range(n_steps))
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA

    df.loc[row_name, cols] = regret_curve
    df = df.sort_index()
    df.to_excel(xlsx)
    print(f"Regret table updated: {xlsx}")


# ── prompt ────────────────────────────────────────────────────────────────────

def build_interactive_prompt(case, template):
    """Fill {n} and {eta} placeholders in an interactive prompt template."""
    n   = len(case["expert_predictions"][0])
    eta = case["learning_rate"]
    return template.replace("{n}", str(n)).replace("{eta}", str(eta))


# ── API call ──────────────────────────────────────────────────────────────────

_RATE_LIMIT_DELAY = {
    # per-call minimum delay (seconds) to stay under RPM limits
    "gemini25_flash": 5, "gemini25_pro": 5,
    "gemini3_flash": 5, "gemini3_pro": 5, "gemini31_pro": 5,
}
_last_call_time = {}

def call_api(messages, model_key, max_retries=10):
    import time
    client = _get_client(model_key)
    kwargs = {"model": MODEL_MAP[model_key], "messages": messages}
    # reasoning models don't support temperature
    if model_key in _REASONING_MODELS:
        kwargs["reasoning_effort"] = "medium"
    elif model_key.startswith("qwen3_14b_think"):
        # Qwen3 thinking mode with budgeted thinking tokens
        # model keys: qwen3_14b_think512, qwen3_14b_think1024, qwen3_14b_think2048
        budget = int(model_key.split("think")[-1]) if model_key != "qwen3_14b" else 1024
        kwargs["temperature"] = 0.6
        kwargs["top_p"] = 0.95
        kwargs["max_tokens"] = budget + 512  # budget + reserve for final JSON
        kwargs["extra_body"] = {
            "top_k": 20, "min_p": 0,
            "thinking_token_budget": budget,
            "chat_template_kwargs": {"enable_thinking": True},
        }
    elif model_key == "qwen3_14b":
        # Qwen3 thinking mode default (budget=1024)
        kwargs["temperature"] = 0.6
        kwargs["top_p"] = 0.95
        kwargs["max_tokens"] = 1536
        kwargs["extra_body"] = {
            "top_k": 20, "min_p": 0,
            "thinking_token_budget": 1024,
            "chat_template_kwargs": {"enable_thinking": True},
        }
    elif model_key == "qwen3_14b_nothink":
        # Qwen3 non-thinking mode
        kwargs["temperature"] = 0.7
        kwargs["top_p"] = 0.8
        kwargs["max_tokens"] = 256
        kwargs["extra_body"] = {
            "top_k": 20, "min_p": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    else:
        kwargs["temperature"] = 0
    # enforce minimum delay between calls for rate-limited providers
    delay = _RATE_LIMIT_DELAY.get(model_key, 0)
    if delay:
        elapsed = time.time() - _last_call_time.get(model_key, 0)
        if elapsed < delay:
            time.sleep(delay - elapsed)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(**kwargs)
            _last_call_time[model_key] = time.time()
            msg       = response.choices[0].message
            content   = msg.content or ""
            # extract reasoning/thinking content
            reasoning = None
            if model_key == "ds_think":
                reasoning = getattr(msg, "reasoning_content", None)
            elif model_key in _REASONING_MODELS:
                reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
            elif model_key in _LOCAL_MODELS:
                # vLLM with --reasoning-parser qwen3: field name varies
                reasoning = (getattr(msg, "reasoning", None)
                             or getattr(msg, "reasoning_content", None))
                # fallback: manual parse if <think> tags in content
                if not reasoning and content and "<think>" in content:
                    import re
                    think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
                    if think_match:
                        reasoning = think_match.group(1).strip()
                        content = content[think_match.end():].strip()
            elif model_key in _GEMINI_MODELS:
                reasoning = getattr(msg, "reasoning_content", None)
            # extract token usage
            usage = {}
            if response.usage:
                u = response.usage
                usage["prompt_tokens"]     = getattr(u, "prompt_tokens", 0) or 0
                usage["completion_tokens"] = getattr(u, "completion_tokens", 0) or 0
                usage["total_tokens"]      = getattr(u, "total_tokens", 0) or 0
                # some providers report reasoning/thinking tokens separately
                if hasattr(u, "completion_tokens_details") and u.completion_tokens_details:
                    d = u.completion_tokens_details
                    usage["reasoning_tokens"] = getattr(d, "reasoning_tokens", 0) or 0
                if hasattr(u, "prompt_tokens_details") and u.prompt_tokens_details:
                    d = u.prompt_tokens_details
                    usage["cached_tokens"] = getattr(d, "cached_tokens", 0) or 0
            return content, reasoning, usage
        except Exception as e:
            if "rate_limit" in str(type(e).__name__).lower() or "429" in str(e) or "resource_exhausted" in str(e).lower():
                wait = min(2 ** attempt + 1, 60)
                print(f"    rate limited, retrying in {wait}s... (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Rate limit exceeded after {max_retries} retries")


# ── token tracking ───────────────────────────────────────────────────────────

# cost per 1M tokens (input, output, reasoning/thinking)
_COST_TABLE = {
    "ds":            {"input": 0.27,  "output": 1.10,  "reasoning": 1.10},
    "ds_think":      {"input": 0.55,  "output": 2.19,  "reasoning": 2.19},
    "gpt4o":         {"input": 2.50,  "output": 10.00, "reasoning": 10.00},
    "gpt4o_mini":    {"input": 0.15,  "output": 0.60,  "reasoning": 0.60},
    "gpt54":         {"input": 2.50,  "output": 15.00, "reasoning": 15.00},
    "gpt54_mini":    {"input": 0.75,  "output": 4.50,  "reasoning": 4.50},
    "gpt54_mini_reason": {"input": 0.75, "output": 4.50, "reasoning": 4.50},
    "gpt54_nano":    {"input": 0.20,  "output": 1.25,  "reasoning": 1.25},
    "o3_mini":       {"input": 1.10,  "output": 4.40,  "reasoning": 4.40},
    "o4_mini":       {"input": 0.55,  "output": 2.20,  "reasoning": 2.20},
    "gemini25_flash": {"input": 0.15, "output": 0.60,  "reasoning": 3.50},
    "gemini25_pro":  {"input": 1.25,  "output": 10.00, "reasoning": 10.00},
    "gemini3_flash": {"input": 0.50,  "output": 3.00,  "reasoning": 3.00},
    "gemini3_pro":   {"input": 1.25,  "output": 10.00, "reasoning": 10.00},
    "gemini31_pro":  {"input": 1.25,  "output": 10.00, "reasoning": 10.00},
    "qwen3_14b":     {"input": 0.00,  "output": 0.00,  "reasoning": 0.00},
}


def _new_token_counter():
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
            "reasoning_tokens": 0, "cached_tokens": 0, "n_calls": 0}


def _accumulate(counter, usage):
    for k in ("prompt_tokens", "completion_tokens", "total_tokens",
              "reasoning_tokens", "cached_tokens"):
        counter[k] += usage.get(k, 0)
    counter["n_calls"] += 1


def _estimate_cost(counter, model_key):
    ct = _COST_TABLE.get(model_key, {"input": 0, "output": 0, "reasoning": 0})
    reasoning = counter["reasoning_tokens"]
    regular_output = counter["completion_tokens"] - reasoning
    cost_input     = counter["prompt_tokens"] / 1e6 * ct["input"]
    cost_output    = regular_output / 1e6 * ct["output"]
    cost_reasoning = reasoning / 1e6 * ct["reasoning"]
    return {
        "input":     round(cost_input, 4),
        "output":    round(cost_output, 4),
        "reasoning": round(cost_reasoning, 4),
        "total":     round(cost_input + cost_output + cost_reasoning, 4),
    }


# ── main run ──────────────────────────────────────────────────────────────────

def run_interactive_once(split, idx, model_key, prompt_name=DEFAULT_PROMPT, plot_weights=False, max_steps=None):
    import time

    if has_existing_run(split, idx, model_key, prompt_name):
        print(f"SKIP: {_label(model_key, prompt_name)} {split}_{idx:03d} already has a run.", flush=True)
        return None

    data = load_data(DATASET_PATH)
    case = get_case(data, split, idx)
    T    = min(case["n_steps"], max_steps) if max_steps else case["n_steps"]
    # truncate case data to T steps
    case = {**case, "n_steps": T,
            "expert_predictions": case["expert_predictions"][:T],
            "true_labels": case["true_labels"][:T],
            "losses": case["losses"][:T]}

    # Build initial message: algorithm setup + step 0 expert predictions
    template        = load_prompt_template(prompt_name)
    setup           = build_interactive_prompt(case, template)
    initial_content = setup + f"\n\nStep 0: expert_predictions = {case['expert_predictions'][0]}"

    messages        = [{"role": "user", "content": initial_content}]
    step_outputs    = []   # parsed dict per step (or None on parse error)
    step_reasonings = []   # reasoning string per step (or None)
    parse_errors    = []   # (step, error_msg, raw_content)
    tokens          = _new_token_counter()
    step_usage_log  = []   # per-step usage dicts

    label = _label(model_key, prompt_name)
    instance_tag = f"{label} {split}_{idx:03d}"
    t_instance_start = time.time()
    t_chunk_start    = t_instance_start

    # ── create run dir NOW so we can stream logs ─────────────────────────────
    run_dir = get_run_dir(split, idx, model_key, prompt_name)
    run_dir.mkdir()
    steps_log_path = run_dir / "steps.jsonl"
    conv_path      = run_dir / "conversation.jsonl"
    usage_path     = run_dir / "token_usage_live.jsonl"

    def _save_step(record):
        with open(steps_log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def _save_conv(msg):
        with open(conv_path, "a") as f:
            f.write(json.dumps(msg) + "\n")

    def _save_usage(usage_record):
        with open(usage_path, "a") as f:
            f.write(json.dumps(usage_record) + "\n")

    # save initial user message
    _save_conv(messages[0])

    print(f"Running {instance_tag} ({T} steps)  → {run_dir}", flush=True)
    _log_timing(f"START {instance_tag} ({T} steps)")

    for t in range(T):
        # ── model predicts for step t ─────────────────────────────────────────
        t_step_start = time.time()
        try:
            content, reasoning, usage = call_api(messages, model_key)
        except Exception as e:
            # API hard failure — save what we have and abort
            _save_step({"split": split, "idx": idx, "model_key": model_key,
                        "prompt": prompt_name, "step": t,
                        "api_error": str(e), "elapsed_s": round(time.time() - t_step_start, 2)})
            print(f"  step {t:2d}: API ERROR: {e}", flush=True)
            _log_timing(f"API ERROR at step {t}: {e}")
            raise

        _accumulate(tokens, usage)
        step_usage_log.append(usage)
        messages.append({"role": "assistant", "content": content})
        step_reasonings.append(reasoning)

        step_record = {"split": split, "idx": idx, "model_key": model_key,
                       "prompt": prompt_name, "step": t,
                       "raw_response": content, "usage": usage,
                       "elapsed_s": round(time.time() - t_step_start, 2)}
        try:
            step_out = parse_model_response(content)
            step_outputs.append(step_out)
            q     = step_out.get("mixture_probability")
            pred  = step_out.get("prediction")
            q_str = f"{q:.4f}" if isinstance(q, (int, float)) else str(q)
            step_record["parsed"] = step_out
            print(f"  step {t:2d}: pred={pred}  q={q_str}  true={case['true_labels'][t]}",
                  flush=(t % 5 == 4))
        except Exception as e:
            print(f"  step {t:2d}: PARSE ERROR: {e}")
            parse_errors.append((t, str(e), content))
            step_outputs.append(None)
            step_record["parse_error"] = str(e)

        if reasoning:
            step_record["reasoning"] = reasoning

        # ── stream: save everything for this step immediately ────────────────
        _save_step(step_record)
        _save_conv({"role": "assistant", "content": content})
        _save_usage({"step": t, "usage": usage, "tokens_cumulative": dict(tokens)})

        # timing: every 10 steps
        if (t + 1) % 10 == 0:
            elapsed_chunk = time.time() - t_chunk_start
            elapsed_total = time.time() - t_instance_start
            msg = (f"  [timer] steps {t-8:>3d}-{t:>3d}: {elapsed_chunk:.1f}s  "
                   f"total: {elapsed_total:.1f}s  ({t+1}/{T})")
            print(msg, flush=True)
            _log_timing(f"{instance_tag}  steps {t-8:>3d}-{t:>3d}: {elapsed_chunk:.1f}s  total: {elapsed_total:.1f}s  ({t+1}/{T})")
            t_chunk_start = time.time()

        true_label = case["true_labels"][t]

        # ── reveal true label; give next step or ask for final weights ────────
        if t < T - 1:
            next_preds = case["expert_predictions"][t + 1]
            user_msg   = (f"True label: {true_label}. "
                          f"Step {t + 1}: expert_predictions = {next_preds}")
        else:
            user_msg = (f'True label: {true_label}. '
                        f'Please output your final updated weights as JSON: {{"weights": [...]}}')
        messages.append({"role": "user", "content": user_msg})
        _save_conv({"role": "user", "content": user_msg})

    # ── get w_T (weights after all T updates) ────────────────────────────────
    final_content, final_reasoning, final_usage = call_api(messages, model_key)
    _accumulate(tokens, final_usage)
    messages.append({"role": "assistant", "content": final_content})
    _save_step({"split": split, "idx": idx, "model_key": model_key,
                "prompt": prompt_name, "step": "final_weights",
                "raw_response": final_content, "usage": final_usage})
    _save_conv({"role": "assistant", "content": final_content})
    _save_usage({"step": "final_weights", "usage": final_usage, "tokens_cumulative": dict(tokens)})

    # ── reconstruct model_output ──────────────────────────────────────────────
    weights_sequence, mixture_probabilities, algorithm_predictions = [], [], []
    for so in step_outputs:
        if so is not None:
            weights_sequence.append(so.get("weights", []))
            mixture_probabilities.append(so.get("mixture_probability", 0.0))
            algorithm_predictions.append(so.get("prediction", 0))
        else:
            weights_sequence.append([])
            mixture_probabilities.append(0.0)
            algorithm_predictions.append(0)

    try:
        final_out = parse_model_response(final_content)
        if "weights" in final_out:
            weights_sequence.append(final_out["weights"])
    except Exception as e:
        print(f"  final weights: PARSE ERROR: {e}")

    model_output = {
        "weights_sequence":      weights_sequence,
        "mixture_probabilities": mixture_probabilities,
        "algorithm_predictions": algorithm_predictions,
    }

    # ── save final results ──────────────────────────────────────────────────
    summary      = analyze_model_output(case, model_output)
    ground_truth = get_ground_truth_outputs(case)

    elapsed = time.time() - t_instance_start
    cost = _estimate_cost(tokens, model_key)

    result = {
        "case_id":  f"{split}_{idx:03d}",
        "meta": {
            "split":     split,
            "idx":       idx,
            "run":       run_dir.name,
            "mode":      "interactive_api",
            "model":     MODEL_MAP[model_key],
            "model_key": model_key,
            "prompt":    prompt_name,
            "n_steps":   case["n_steps"],
            "elapsed_s": round(elapsed, 1),
        },
        "token_usage": tokens,
        "estimated_cost": cost,
        "analysis": summary,
        "input": {
            "n":                  len(case["expert_predictions"][0]),
            "T":                  case["n_steps"],
            "eta":                case["learning_rate"],
            "expert_predictions": case["expert_predictions"],
            "true_labels":        case["true_labels"],
            "losses":             case["losses"],
        },
        "ground_truth": ground_truth,
        "response":     model_output,
    }
    if parse_errors:
        result["meta"]["parse_errors"] = parse_errors

    (run_dir / "result.json").write_text(json.dumps(result, indent=2))
    (run_dir / "conversation.json").write_text(json.dumps(messages, indent=2))
    (run_dir / "step_outputs.json").write_text(json.dumps(step_outputs, indent=2))
    (run_dir / "token_usage.json").write_text(json.dumps(
        {"total": tokens, "cost": cost, "per_step": step_usage_log}, indent=2))

    # reasoning (R1 only)
    blocks = [f"=== Step {i} ===\n{r}" for i, r in enumerate(step_reasonings) if r]
    if final_reasoning:
        blocks.append(f"=== Final weights ===\n{final_reasoning}")
    if blocks:
        (run_dir / "reasoning.txt").write_text("\n\n".join(blocks))

    # report notebook + plots
    _write_report_notebook(run_dir / "report.ipynb", split, idx, run_dir)
    plot_regret_comparison(case, model_output, save_path=run_dir / "plot_regret.png")
    if plot_weights:
        plot_weight_comparison(case, model_output, save_path=run_dir / "plot_weights.png")
    plot_cumulative_loss(summary, case=case, model_output=model_output,
                         save_path=run_dir / "plot_loss.png")

    # ── print summary ─────────────────────────────────────────────────────────
    w = summary["weights"]
    p = summary["mixture_probabilities"]
    y = summary["algorithm_predictions"]
    print(f"\nSaved to: {run_dir}/")
    print()
    print("=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    print(f"  weights.check_norm    : {w['check_norm']}")
    print(f"  weights.mean_ell1     : {w['mean_ell1']:.4f}  std={w['std_ell1']:.4f}")
    print()
    print(f"  prob.check_compute    : {p['check_compute']}")
    print(f"  prob.mean_abs_error   : {p['mean_abs_error']:.4f}  std={p['std_abs_error']:.4f}")
    print()
    print(f"  pred.check_pred       : {y['check_pred']}")
    print(f"  pred.accuracy         : {y['accuracy']:.3f}")
    if parse_errors:
        print(f"\nWarning: {len(parse_errors)} parse error(s) at steps "
              f"{[e[0] for e in parse_errors]}")

    msg = f"DONE {instance_tag}  total: {elapsed:.1f}s  ({elapsed/T:.2f}s/step)"
    cost_msg = (f"  tokens: {tokens['prompt_tokens']:,} in + {tokens['completion_tokens']:,} out "
                f"(reasoning: {tokens['reasoning_tokens']:,})  cost: ${cost['total']:.4f}")
    print(f"\n  [timer] {msg}", flush=True)
    print(cost_msg, flush=True)
    _log_timing(msg)
    _log_timing(cost_msg)

    _update_regret_table(model_key, prompt_name, split, idx,
                         summary["algorithm_predictions"]["model_regret"])
    return run_dir


# ── interactive online (prediction-only) ─────────────────────────────────────

ONLINE_PROMPTS  = set()  # legacy, all online prompts now use two-turn or nonote
WEATHER_PROMPTS = {"interactive_weather", "interactive_weather_no_hint",
                    "interactive_weather_v2", "interactive_weather_no_hint_v2"}  # weather-style prompts (sunny/rainy)

# rainy ↔ 0, sunny ↔ 1
_TO_WEATHER   = {0: "rainy", 1: "sunny"}
_FROM_WEATHER = {"rainy": 0, "sunny": 1}


def _format_preds(expert_preds_t, weather):
    """Format one step's expert predictions for the user message."""
    if weather:
        return [_TO_WEATHER[p] for p in expert_preds_t]
    return expert_preds_t


def _format_label(true_label, weather):
    if weather:
        return _TO_WEATHER[true_label]
    return true_label


def _parse_pred(step_out, weather):
    """Extract a numeric prediction (0/1) from the parsed model JSON."""
    raw = step_out.get("prediction", 0)
    if weather and isinstance(raw, str):
        return _FROM_WEATHER.get(raw.lower(), 0)
    return raw


def run_interactive_online_once(split, idx, model_key, prompt_name="interactive_online", max_steps=None):
    import time

    if has_existing_run(split, idx, model_key, prompt_name):
        print(f"SKIP: {_label(model_key, prompt_name)} {split}_{idx:03d} already has a run.", flush=True)
        return None

    data = load_data(DATASET_PATH)
    case = get_case(data, split, idx)
    T    = min(case["n_steps"], max_steps) if max_steps else case["n_steps"]
    case = {**case, "n_steps": T,
            "expert_predictions": case["expert_predictions"][:T],
            "true_labels": case["true_labels"][:T],
            "losses": case["losses"][:T]}
    weather = prompt_name in WEATHER_PROMPTS

    template        = load_prompt_template(prompt_name)
    setup           = build_interactive_prompt(case, template)
    preds_0         = _format_preds(case["expert_predictions"][0], weather)
    field_name      = "opinions" if weather else "expert_predictions"
    initial_content = setup + f"\n\nDay 1: {field_name} = {preds_0}" if weather \
                      else setup + f"\n\nStep 0: expert_predictions = {preds_0}"

    messages        = [{"role": "user", "content": initial_content}]
    predictions     = []
    step_reasonings = []
    parse_errors    = []
    tokens          = _new_token_counter()
    step_usage_log  = []

    label = _label(model_key, prompt_name)
    instance_tag = f"{label} {split}_{idx:03d}"
    t_instance_start = time.time()
    t_chunk_start    = t_instance_start

    # ── create run dir NOW so we can stream logs ─────────────────────────────
    run_dir = get_run_dir(split, idx, model_key, prompt_name)
    run_dir.mkdir()
    steps_log_path = run_dir / "steps.jsonl"
    conv_path      = run_dir / "conversation.jsonl"
    usage_path     = run_dir / "token_usage_live.jsonl"

    def _save_step(record):
        with open(steps_log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def _save_conv(msg):
        with open(conv_path, "a") as f:
            f.write(json.dumps(msg) + "\n")

    def _save_usage(usage_record):
        with open(usage_path, "a") as f:
            f.write(json.dumps(usage_record) + "\n")

    # save initial user message
    _save_conv(messages[0])

    print(f"Running {instance_tag} ({T} steps)  → {run_dir}", flush=True)
    _log_timing(f"START {instance_tag} ({T} steps)")

    for t in range(T):
        t_step_start = time.time()
        try:
            content, reasoning, usage = call_api(messages, model_key)
        except Exception as e:
            _save_step({"split": split, "idx": idx, "model_key": model_key,
                        "prompt": prompt_name, "step": t,
                        "api_error": str(e), "elapsed_s": round(time.time() - t_step_start, 2)})
            print(f"  step {t:2d}: API ERROR: {e}", flush=True)
            _log_timing(f"API ERROR at step {t}: {e}")
            raise

        _accumulate(tokens, usage)
        step_usage_log.append(usage)
        messages.append({"role": "assistant", "content": content})
        step_reasonings.append(reasoning)

        step_record = {"split": split, "idx": idx, "model_key": model_key,
                       "prompt": prompt_name, "step": t,
                       "raw_response": content, "usage": usage,
                       "elapsed_s": round(time.time() - t_step_start, 2)}
        try:
            step_out = parse_model_response(content)
            pred = _parse_pred(step_out, weather)
            predictions.append(pred)
            raw_pred = step_out.get("prediction", pred)
            true_lbl = _format_label(case["true_labels"][t], weather)
            step_record["parsed"] = step_out
            print(f"  step {t:2d}: pred={raw_pred}  true={true_lbl}",
                  flush=(t % 5 == 4))
        except Exception as e:
            print(f"  step {t:2d}: PARSE ERROR: {e}")
            parse_errors.append((t, str(e), content))
            predictions.append(0)
            step_record["parse_error"] = str(e)

        if reasoning:
            step_record["reasoning"] = reasoning

        # ── stream: save everything for this step immediately ────────────────
        _save_step(step_record)
        _save_conv({"role": "assistant", "content": content})
        _save_usage({"step": t, "usage": usage, "tokens_cumulative": dict(tokens)})

        # timing: every 10 steps
        if (t + 1) % 10 == 0:
            elapsed_chunk = time.time() - t_chunk_start
            elapsed_total = time.time() - t_instance_start
            msg = (f"  [timer] steps {t-8:>3d}-{t:>3d}: {elapsed_chunk:.1f}s  "
                   f"total: {elapsed_total:.1f}s  ({t+1}/{T})")
            print(msg, flush=True)
            _log_timing(f"{instance_tag}  steps {t-8:>3d}-{t:>3d}: {elapsed_chunk:.1f}s  total: {elapsed_total:.1f}s  ({t+1}/{T})")
            t_chunk_start = time.time()

        true_label = _format_label(case["true_labels"][t], weather)
        if t < T - 1:
            next_preds = _format_preds(case["expert_predictions"][t + 1], weather)
            if weather:
                user_msg = (f"Actual weather: {true_label}. "
                            f"Day {t + 2}: opinions = {next_preds}")
            else:
                user_msg = (f"True label: {true_label}. "
                            f"Step {t + 1}: expert_predictions = {next_preds}")
        else:
            if weather:
                user_msg = f"Actual weather: {true_label}. All days complete."
            else:
                user_msg = f"True label: {true_label}. All steps complete."
        messages.append({"role": "user", "content": user_msg})
        _save_conv({"role": "user", "content": user_msg})

    # ── save final results ──────────────────────────────────────────────────
    analysis     = analyze_online_output(case, predictions)
    ground_truth = get_ground_truth_outputs(case)

    elapsed = time.time() - t_instance_start
    cost = _estimate_cost(tokens, model_key)

    result = {
        "case_id":  f"{split}_{idx:03d}",
        "meta": {
            "split":     split,
            "idx":       idx,
            "run":       run_dir.name,
            "mode":      "interactive_online",
            "model":     MODEL_MAP[model_key],
            "model_key": model_key,
            "prompt":    prompt_name,
            "n_steps":   T,
            "elapsed_s": round(elapsed, 1),
        },
        "token_usage": tokens,
        "estimated_cost": cost,
        "analysis":     analysis,
        "input": {
            "n":                  len(case["expert_predictions"][0]),
            "T":                  T,
            "eta":                case["learning_rate"],
            "expert_predictions": case["expert_predictions"],
            "true_labels":        case["true_labels"],
            "losses":             case["losses"],
        },
        "ground_truth": ground_truth,
        "response":     {"predictions": predictions},
    }
    if parse_errors:
        result["meta"]["parse_errors"] = parse_errors

    (run_dir / "result.json").write_text(json.dumps(result, indent=2))
    (run_dir / "conversation.json").write_text(json.dumps(messages, indent=2))
    (run_dir / "token_usage.json").write_text(json.dumps(
        {"total": tokens, "cost": cost, "per_step": step_usage_log}, indent=2))

    # reasoning (R1 only)
    blocks = [f"=== Step {i} ===\n{r}" for i, r in enumerate(step_reasonings) if r]
    if blocks:
        (run_dir / "reasoning.txt").write_text("\n\n".join(blocks))

    # plots (no weight plot for online mode)
    plot_online_regret(analysis, save_path=run_dir / "plot_regret.png")
    plot_online_loss(analysis, save_path=run_dir / "plot_loss.png")

    # ── print summary ────────────────────────────────────────────────────────
    print(f"\nSaved to: {run_dir}/")
    print()
    print("=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    print(f"  accuracy          : {analysis['accuracy']:.3f}")
    print(f"  model final loss  : {analysis['model_cum_loss'][-1]}")
    print(f"  MW final loss     : {analysis['true_cum_loss'][-1]}")
    print(f"  best expert loss  : {analysis['best_expert_cum'][-1]}")
    print(f"  model final regret: {analysis['regret_curve'][-1]}")
    print(f"  MW final regret   : {analysis['true_regret_curve'][-1]}")
    if parse_errors:
        print(f"\nWarning: {len(parse_errors)} parse error(s) at steps "
              f"{[e[0] for e in parse_errors]}")

    msg = f"DONE {instance_tag}  total: {elapsed:.1f}s  ({elapsed/T:.2f}s/step)"
    cost_msg = (f"  tokens: {tokens['prompt_tokens']:,} in + {tokens['completion_tokens']:,} out "
                f"(reasoning: {tokens['reasoning_tokens']:,})  cost: ${cost['total']:.4f}")
    print(f"\n  [timer] {msg}", flush=True)
    print(cost_msg, flush=True)
    _log_timing(msg)
    _log_timing(cost_msg)

    _update_regret_table(model_key, prompt_name, split, idx, analysis["regret_curve"])
    return run_dir


# ── two-turn weather protocol (stateless, note-passing) ─────────────────────

TWO_TURN_PROMPTS = {"interactive_weather", "interactive_online",
                    "interactive_weather_no_hint", "interactive_online",
                    "interactive_online_nohint",
                    "interactive_weather_v2", "interactive_weather_no_hint_v2",
                    "interactive_online_v2", "interactive_online_nohint_v2"}

_TO_WEATHER   = {0: "rainy", 1: "sunny"}
_FROM_WEATHER_V2 = {"rainy": 0, "sunny": 1}

_EXPERT_NAMES = ["Expert_A", "Expert_B", "Expert_C", "Expert_D"]


def run_two_turn(split, idx, model_key, prompt_name, max_steps=None):
    import time

    if has_existing_run(split, idx, model_key, prompt_name):
        print(f"SKIP: {_label(model_key, prompt_name)} {split}_{idx:03d} already has a run.", flush=True)
        return None

    data = load_data(DATASET_PATH)
    case = get_case(data, split, idx)
    T    = min(case["n_steps"], max_steps) if max_steps else case["n_steps"]
    case = {**case, "n_steps": T,
            "expert_predictions": case["expert_predictions"][:T],
            "true_labels": case["true_labels"][:T],
            "losses": case["losses"][:T]}

    system_prompt = load_prompt_template(prompt_name)

    tokens          = _new_token_counter()
    step_usage_log  = []
    predictions     = []
    notes           = [""]  # note_0 = empty
    parse_errors    = []

    label = _label(model_key, prompt_name)
    instance_tag = f"{label} {split}_{idx:03d}"
    t_instance_start = time.time()
    t_chunk_start    = t_instance_start

    # create run dir
    run_dir = get_run_dir(split, idx, model_key, prompt_name)
    run_dir.mkdir()
    steps_log_path = run_dir / "steps.jsonl"
    usage_path     = run_dir / "token_usage_live.jsonl"

    def _save_step(record):
        with open(steps_log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def _save_usage(usage_record):
        with open(usage_path, "a") as f:
            f.write(json.dumps(usage_record) + "\n")

    print(f"Running {instance_tag} ({T} steps, two-turn)  → {run_dir}", flush=True)
    _log_timing(f"START {instance_tag} ({T} steps, two-turn)")

    note = ""
    for t in range(T):
        t_step_start = time.time()
        weather = prompt_name in WEATHER_PROMPTS
        raw_preds = case["expert_predictions"][t]
        raw_label = case["true_labels"][t]

        if weather:
            ep_dict = {_EXPERT_NAMES[i]: _TO_WEATHER[p] for i, p in enumerate(raw_preds)}
            true_val = _TO_WEATHER[raw_label]
            label_key = "actual_weather"
        else:
            ep_dict = {_EXPERT_NAMES[i]: p for i, p in enumerate(raw_preds)}
            true_val = raw_label
            label_key = "true_label"

        # ── prediction turn ───────────────────────────────────────────────────
        turn_type_pred = "forecast" if weather else "prediction"
        forecast_input = json.dumps({
            "turn_type": turn_type_pred,
            "note": note,
            **ep_dict,
        })
        forecast_msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": forecast_input},
        ]

        try:
            fc_content, fc_reasoning, fc_usage = call_api(forecast_msgs, model_key)
        except Exception as e:
            _save_step({"split": split, "idx": idx, "model_key": model_key,
                        "prompt": prompt_name, "step": t, "turn": "forecast",
                        "api_error": str(e), "elapsed_s": round(time.time() - t_step_start, 2)})
            print(f"  step {t:2d}: FORECAST API ERROR: {e}", flush=True)
            _log_timing(f"API ERROR at step {t} forecast: {e}")
            raise

        _accumulate(tokens, fc_usage)
        step_usage_log.append({"forecast": fc_usage})

        # parse forecast
        pred_weather = None
        fc_record = {"split": split, "idx": idx, "model_key": model_key,
                     "prompt": prompt_name, "step": t, "turn": "forecast",
                     "note_in": note, "raw_response": fc_content,
                     "usage": fc_usage, "elapsed_s": round(time.time() - t_step_start, 2)}
        if fc_reasoning:
            fc_record["reasoning"] = fc_reasoning

        try:
            fc_parsed = parse_model_response(fc_content)
            raw_pred = fc_parsed.get("prediction", "")
            if weather and isinstance(raw_pred, str):
                pred_num = _FROM_WEATHER_V2.get(raw_pred.lower(), 0)
            else:
                pred_num = int(raw_pred) if raw_pred in (0, 1, "0", "1") else 0
            predictions.append(pred_num)
            fc_record["parsed"] = fc_parsed
            display_pred = raw_pred if weather else pred_num
            display_true = true_val
            print(f"  step {t:2d}: pred={display_pred}  true={display_true}",
                  flush=(t % 5 == 4))
        except Exception as e:
            print(f"  step {t:2d}: FORECAST PARSE ERROR: {e}")
            parse_errors.append((t, "forecast", str(e), fc_content))
            predictions.append(0)
            fc_record["parse_error"] = str(e)

        _save_step(fc_record)

        # ── feedback turn (note version) ─────────────────────────────────────
        t_update_start = time.time()
        update_input = json.dumps({
            "turn_type": "feedback",
            "note": note,
            **ep_dict,
            label_key: true_val,
        })
        update_msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": update_input},
        ]

        try:
            up_content, up_reasoning, up_usage = call_api(update_msgs, model_key)
        except Exception as e:
            _save_step({"split": split, "idx": idx, "model_key": model_key,
                        "prompt": prompt_name, "step": t, "turn": "feedback",
                        "api_error": str(e), "elapsed_s": round(time.time() - t_update_start, 2)})
            print(f"  step {t:2d}: FEEDBACK API ERROR: {e}", flush=True)
            _log_timing(f"API ERROR at step {t} feedback: {e}")
            raise

        _accumulate(tokens, up_usage)
        step_usage_log[-1]["feedback"] = up_usage

        up_record = {"split": split, "idx": idx, "model_key": model_key,
                     "prompt": prompt_name, "step": t, "turn": "feedback",
                     "note_in": note, "raw_response": up_content,
                     "usage": up_usage, "elapsed_s": round(time.time() - t_update_start, 2)}
        if up_reasoning:
            up_record["reasoning"] = up_reasoning

        try:
            up_parsed = parse_model_response(up_content)
            new_note = up_parsed.get("note", "")
            up_record["parsed"] = up_parsed
            up_record["note_out"] = new_note
        except Exception as e:
            print(f"  step {t:2d}: FEEDBACK PARSE ERROR: {e}")
            parse_errors.append((t, "feedback", str(e), up_content))
            new_note = note  # keep old note on failure
            up_record["parse_error"] = str(e)

        _save_step(up_record)
        _save_usage({"step": t, "forecast": fc_usage, "feedback": up_usage,
                     "tokens_cumulative": dict(tokens)})

        note = new_note
        notes.append(note)

        # timing
        if (t + 1) % 10 == 0:
            elapsed_chunk = time.time() - t_chunk_start
            elapsed_total = time.time() - t_instance_start
            msg = (f"  [timer] steps {t-8:>3d}-{t:>3d}: {elapsed_chunk:.1f}s  "
                   f"total: {elapsed_total:.1f}s  ({t+1}/{T})")
            print(msg, flush=True)
            _log_timing(f"{instance_tag}  steps {t-8:>3d}-{t:>3d}: {elapsed_chunk:.1f}s  total: {elapsed_total:.1f}s  ({t+1}/{T})")
            t_chunk_start = time.time()

    # ── save final results ───────────────────────────────────────────────────
    elapsed = time.time() - t_instance_start
    cost = _estimate_cost(tokens, model_key)
    analysis = analyze_online_output(case, predictions)
    ground_truth = get_ground_truth_outputs(case)

    result = {
        "case_id":  f"{split}_{idx:03d}",
        "meta": {
            "split":     split,
            "idx":       idx,
            "run":       run_dir.name,
            "mode":      "two_turn_weather",
            "model":     MODEL_MAP[model_key],
            "model_key": model_key,
            "prompt":    prompt_name,
            "n_steps":   T,
            "elapsed_s": round(elapsed, 1),
        },
        "token_usage": tokens,
        "estimated_cost": cost,
        "analysis":     analysis,
        "input": {
            "n":                  len(case["expert_predictions"][0]),
            "T":                  T,
            "eta":                case["learning_rate"],
            "expert_predictions": case["expert_predictions"],
            "true_labels":        case["true_labels"],
            "losses":             case["losses"],
        },
        "ground_truth": ground_truth,
        "response":     {"predictions": predictions, "notes": notes},
    }
    if parse_errors:
        result["meta"]["parse_errors"] = parse_errors

    (run_dir / "result.json").write_text(json.dumps(result, indent=2))
    (run_dir / "token_usage.json").write_text(json.dumps(
        {"total": tokens, "cost": cost, "per_step": step_usage_log}, indent=2))

    # plots
    plot_online_regret(analysis, save_path=run_dir / "plot_regret.png")
    plot_online_loss(analysis, save_path=run_dir / "plot_loss.png")

    # summary
    print(f"\nSaved to: {run_dir}/")
    print()
    print("=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    print(f"  accuracy          : {analysis['accuracy']:.3f}")
    print(f"  model final loss  : {analysis['model_cum_loss'][-1]}")
    print(f"  MW final loss     : {analysis['true_cum_loss'][-1]}")
    print(f"  best expert loss  : {analysis['best_expert_cum'][-1]}")
    print(f"  model final regret: {analysis['regret_curve'][-1]}")
    print(f"  MW final regret   : {analysis['true_regret_curve'][-1]}")
    if parse_errors:
        print(f"\nWarning: {len(parse_errors)} parse error(s)")

    msg = f"DONE {instance_tag}  total: {elapsed:.1f}s  ({elapsed/T:.2f}s/step)"
    cost_msg = (f"  tokens: {tokens['prompt_tokens']:,} in + {tokens['completion_tokens']:,} out "
                f"(reasoning: {tokens['reasoning_tokens']:,})  cost: ${cost['total']:.4f}")
    print(f"\n  [timer] {msg}", flush=True)
    print(cost_msg, flush=True)
    _log_timing(msg)
    _log_timing(cost_msg)

    _update_regret_table(model_key, prompt_name, split, idx, analysis["regret_curve"])
    return run_dir


# ── nonote protocol (multi-turn, feedback appended as user message) ──────────

NONOTE_PROMPTS = {"interactive_weather_nonote", "interactive_weather_nonote_nohint",
                  "interactive_online_nonote", "interactive_online_nonote_nohint",
                  "interactive_weather_nonote_v2", "interactive_weather_nonote_nohint_v2",
                  "interactive_online_nonote_v2", "interactive_online_nonote_nohint_v2"}
NONOTE_WEATHER = {"interactive_weather_nonote", "interactive_weather_nonote_nohint",
                  "interactive_weather_nonote_v2", "interactive_weather_nonote_nohint_v2"}


def run_nonote(split, idx, model_key, prompt_name, max_steps=None):
    import time

    if has_existing_run(split, idx, model_key, prompt_name):
        print(f"SKIP: {_label(model_key, prompt_name)} {split}_{idx:03d} already has a run.", flush=True)
        return None

    data = load_data(DATASET_PATH)
    case = get_case(data, split, idx)
    T    = min(case["n_steps"], max_steps) if max_steps else case["n_steps"]
    case = {**case, "n_steps": T,
            "expert_predictions": case["expert_predictions"][:T],
            "true_labels": case["true_labels"][:T],
            "losses": case["losses"][:T]}

    system_prompt = load_prompt_template(prompt_name)
    weather = prompt_name in NONOTE_WEATHER

    tokens          = _new_token_counter()
    step_usage_log  = []
    predictions     = []
    parse_errors    = []

    label = _label(model_key, prompt_name)
    instance_tag = f"{label} {split}_{idx:03d}"
    t_instance_start = time.time()
    t_chunk_start    = t_instance_start

    run_dir = get_run_dir(split, idx, model_key, prompt_name)
    run_dir.mkdir()
    steps_log_path = run_dir / "steps.jsonl"
    conv_path      = run_dir / "conversation.jsonl"
    usage_path     = run_dir / "token_usage_live.jsonl"

    def _save_step(record):
        with open(steps_log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def _save_conv(msg):
        with open(conv_path, "a") as f:
            f.write(json.dumps(msg) + "\n")

    def _save_usage(usage_record):
        with open(usage_path, "a") as f:
            f.write(json.dumps(usage_record) + "\n")

    # conversation history (multi-turn)
    messages = [{"role": "system", "content": system_prompt}]
    _save_conv(messages[0])

    print(f"Running {instance_tag} ({T} steps, nonote)  → {run_dir}", flush=True)
    _log_timing(f"START {instance_tag} ({T} steps, nonote)")

    for t in range(T):
        t_step_start = time.time()
        raw_preds = case["expert_predictions"][t]
        raw_label = case["true_labels"][t]

        if weather:
            ep_dict = {_EXPERT_NAMES[i]: _TO_WEATHER[p] for i, p in enumerate(raw_preds)}
            true_val = _TO_WEATHER[raw_label]
            turn_type_pred = "forecast"
            label_key = "actual_weather"
        else:
            ep_dict = {_EXPERT_NAMES[i]: p for i, p in enumerate(raw_preds)}
            true_val = raw_label
            turn_type_pred = "prediction"
            label_key = "true_label"

        # ── forecast/prediction turn ─────────────────────────────────────────
        forecast_input = json.dumps({"turn_type": turn_type_pred, **ep_dict})
        messages.append({"role": "user", "content": forecast_input})
        _save_conv({"role": "user", "content": forecast_input})

        try:
            fc_content, fc_reasoning, fc_usage = call_api(messages, model_key)
        except Exception as e:
            _save_step({"split": split, "idx": idx, "model_key": model_key,
                        "prompt": prompt_name, "step": t, "turn": turn_type_pred,
                        "api_error": str(e), "elapsed_s": round(time.time() - t_step_start, 2)})
            print(f"  step {t:2d}: API ERROR: {e}", flush=True)
            raise

        _accumulate(tokens, fc_usage)
        step_usage_log.append(fc_usage)
        messages.append({"role": "assistant", "content": fc_content})
        _save_conv({"role": "assistant", "content": fc_content})

        step_record = {"split": split, "idx": idx, "model_key": model_key,
                       "prompt": prompt_name, "step": t, "turn": turn_type_pred,
                       "raw_response": fc_content, "usage": fc_usage,
                       "elapsed_s": round(time.time() - t_step_start, 2)}
        if fc_reasoning:
            step_record["reasoning"] = fc_reasoning

        try:
            fc_parsed = parse_model_response(fc_content)
            raw_pred = fc_parsed.get("prediction", "")
            if weather and isinstance(raw_pred, str):
                pred_num = _FROM_WEATHER_V2.get(raw_pred.lower(), 0)
            else:
                pred_num = int(raw_pred) if raw_pred in (0, 1, "0", "1") else 0
            predictions.append(pred_num)
            step_record["parsed"] = fc_parsed
            display_pred = raw_pred if weather else pred_num
            print(f"  step {t:2d}: pred={display_pred}  true={true_val}",
                  flush=(t % 5 == 4))
        except Exception as e:
            print(f"  step {t:2d}: PARSE ERROR: {e}")
            parse_errors.append((t, str(e), fc_content))
            predictions.append(0)
            step_record["parse_error"] = str(e)

        _save_step(step_record)
        _save_usage({"step": t, "usage": fc_usage, "tokens_cumulative": dict(tokens)})

        # ── feedback turn (no model call, just append to history) ────────────
        feedback_input = json.dumps({"turn_type": "feedback", label_key: true_val})
        messages.append({"role": "user", "content": feedback_input})
        _save_conv({"role": "user", "content": feedback_input})

        # timing
        if (t + 1) % 10 == 0:
            elapsed_chunk = time.time() - t_chunk_start
            elapsed_total = time.time() - t_instance_start
            msg = (f"  [timer] steps {t-8:>3d}-{t:>3d}: {elapsed_chunk:.1f}s  "
                   f"total: {elapsed_total:.1f}s  ({t+1}/{T})")
            print(msg, flush=True)
            _log_timing(f"{instance_tag}  steps {t-8:>3d}-{t:>3d}: {elapsed_chunk:.1f}s  total: {elapsed_total:.1f}s  ({t+1}/{T})")
            t_chunk_start = time.time()

    # ── save final results ───────────────────────────────────────────────────
    elapsed = time.time() - t_instance_start
    cost = _estimate_cost(tokens, model_key)
    analysis = analyze_online_output(case, predictions)
    ground_truth = get_ground_truth_outputs(case)

    result = {
        "case_id":  f"{split}_{idx:03d}",
        "meta": {
            "split":     split,
            "idx":       idx,
            "run":       run_dir.name,
            "mode":      "nonote",
            "model":     MODEL_MAP[model_key],
            "model_key": model_key,
            "prompt":    prompt_name,
            "n_steps":   T,
            "elapsed_s": round(elapsed, 1),
        },
        "token_usage": tokens,
        "estimated_cost": cost,
        "analysis":     analysis,
        "input": {
            "n":                  len(case["expert_predictions"][0]),
            "T":                  T,
            "eta":                case["learning_rate"],
            "expert_predictions": case["expert_predictions"],
            "true_labels":        case["true_labels"],
            "losses":             case["losses"],
        },
        "ground_truth": ground_truth,
        "response":     {"predictions": predictions},
    }
    if parse_errors:
        result["meta"]["parse_errors"] = parse_errors

    (run_dir / "result.json").write_text(json.dumps(result, indent=2))
    (run_dir / "token_usage.json").write_text(json.dumps(
        {"total": tokens, "cost": cost, "per_step": step_usage_log}, indent=2))

    plot_online_regret(analysis, save_path=run_dir / "plot_regret.png")
    plot_online_loss(analysis, save_path=run_dir / "plot_loss.png")

    print(f"\nSaved to: {run_dir}/")
    print()
    print("=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    print(f"  accuracy          : {analysis['accuracy']:.3f}")
    print(f"  model final loss  : {analysis['model_cum_loss'][-1]}")
    print(f"  MW final loss     : {analysis['true_cum_loss'][-1]}")
    print(f"  best expert loss  : {analysis['best_expert_cum'][-1]}")
    print(f"  model final regret: {analysis['regret_curve'][-1]}")
    print(f"  MW final regret   : {analysis['true_regret_curve'][-1]}")
    if parse_errors:
        print(f"\nWarning: {len(parse_errors)} parse error(s)")

    msg = f"DONE {instance_tag}  total: {elapsed:.1f}s  ({elapsed/T:.2f}s/step)"
    cost_msg = (f"  tokens: {tokens['prompt_tokens']:,} in + {tokens['completion_tokens']:,} out "
                f"(reasoning: {tokens['reasoning_tokens']:,})  cost: ${cost['total']:.4f}")
    print(f"\n  [timer] {msg}", flush=True)
    print(cost_msg, flush=True)
    _log_timing(msg)
    _log_timing(cost_msg)

    _update_regret_table(model_key, prompt_name, split, idx, analysis["regret_curve"])
    return run_dir


# ── note-with-history (multi-turn + note) ────────────────────────────────────

NOTE_HISTORY_PROMPTS = {
    "interactive_weather_v2_notehist", "interactive_weather_no_hint_v2_notehist",
    "interactive_online_v2_notehist", "interactive_online_nohint_v2_notehist",
}
NOTE_HISTORY_WEATHER = {
    "interactive_weather_v2_notehist", "interactive_weather_no_hint_v2_notehist",
}
# Map notehist prompt names to the actual prompt template files (same as note prompts)
_NOTEHIST_TO_TEMPLATE = {
    "interactive_weather_v2_notehist": "interactive_weather_v2",
    "interactive_weather_no_hint_v2_notehist": "interactive_weather_no_hint_v2",
    "interactive_online_v2_notehist": "interactive_online_v2",
    "interactive_online_nohint_v2_notehist": "interactive_online_nohint_v2",
}


def run_note_with_history(split, idx, model_key, prompt_name, max_steps=None):
    """Like run_nonote (multi-turn with full history) but the model also
    produces a note on each feedback turn. Combines conversation history
    with explicit note state."""
    import time

    if has_existing_run(split, idx, model_key, prompt_name):
        print(f"SKIP: {_label(model_key, prompt_name)} {split}_{idx:03d} already has a run.", flush=True)
        return None

    data = load_data(DATASET_PATH)
    case = get_case(data, split, idx)
    T    = min(case["n_steps"], max_steps) if max_steps else case["n_steps"]
    case = {**case, "n_steps": T,
            "expert_predictions": case["expert_predictions"][:T],
            "true_labels": case["true_labels"][:T],
            "losses": case["losses"][:T]}

    # Use the corresponding note template (same prompt file as two-turn note)
    template_name = _NOTEHIST_TO_TEMPLATE.get(prompt_name, prompt_name)
    system_prompt = load_prompt_template(template_name)
    weather = prompt_name in NOTE_HISTORY_WEATHER

    tokens          = _new_token_counter()
    step_usage_log  = []
    predictions     = []
    notes           = [""]
    parse_errors    = []

    label = _label(model_key, prompt_name)
    instance_tag = f"{label} {split}_{idx:03d}"
    t_instance_start = time.time()
    t_chunk_start    = t_instance_start

    run_dir = get_run_dir(split, idx, model_key, prompt_name)
    run_dir.mkdir()
    steps_log_path = run_dir / "steps.jsonl"
    conv_path      = run_dir / "conversation.jsonl"
    usage_path     = run_dir / "token_usage_live.jsonl"

    def _save_step(record):
        with open(steps_log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def _save_conv(msg):
        with open(conv_path, "a") as f:
            f.write(json.dumps(msg) + "\n")

    def _save_usage(usage_record):
        with open(usage_path, "a") as f:
            f.write(json.dumps(usage_record) + "\n")

    # conversation history (multi-turn, like nonote)
    messages = [{"role": "system", "content": system_prompt}]
    _save_conv(messages[0])

    print(f"Running {instance_tag} ({T} steps, note-with-history)  → {run_dir}", flush=True)
    _log_timing(f"START {instance_tag} ({T} steps, note-with-history)")

    note = ""
    for t in range(T):
        t_step_start = time.time()
        raw_preds = case["expert_predictions"][t]
        raw_label = case["true_labels"][t]

        if weather:
            ep_dict = {_EXPERT_NAMES[i]: _TO_WEATHER[p] for i, p in enumerate(raw_preds)}
            true_val = _TO_WEATHER[raw_label]
            turn_type_pred = "forecast"
            label_key = "actual_weather"
        else:
            ep_dict = {_EXPERT_NAMES[i]: p for i, p in enumerate(raw_preds)}
            true_val = raw_label
            turn_type_pred = "prediction"
            label_key = "true_label"

        # ── forecast/prediction turn (with note in input) ────────────────────
        forecast_input = json.dumps({
            "turn_type": turn_type_pred,
            "note": note,
            **ep_dict,
        })
        messages.append({"role": "user", "content": forecast_input})
        _save_conv({"role": "user", "content": forecast_input})

        try:
            fc_content, fc_reasoning, fc_usage = call_api(messages, model_key)
        except Exception as e:
            _save_step({"split": split, "idx": idx, "model_key": model_key,
                        "prompt": prompt_name, "step": t, "turn": turn_type_pred,
                        "api_error": str(e), "elapsed_s": round(time.time() - t_step_start, 2)})
            print(f"  step {t:2d}: API ERROR: {e}", flush=True)
            raise

        _accumulate(tokens, fc_usage)
        step_usage_log.append({"forecast": fc_usage})
        messages.append({"role": "assistant", "content": fc_content})
        _save_conv({"role": "assistant", "content": fc_content})

        step_record = {"split": split, "idx": idx, "model_key": model_key,
                       "prompt": prompt_name, "step": t, "turn": turn_type_pred,
                       "note_in": note, "raw_response": fc_content, "usage": fc_usage,
                       "elapsed_s": round(time.time() - t_step_start, 2)}
        if fc_reasoning:
            step_record["reasoning"] = fc_reasoning

        try:
            fc_parsed = parse_model_response(fc_content)
            raw_pred = fc_parsed.get("prediction", "")
            if weather and isinstance(raw_pred, str):
                pred_num = _FROM_WEATHER_V2.get(raw_pred.lower(), 0)
            else:
                pred_num = int(raw_pred) if raw_pred in (0, 1, "0", "1") else 0
            predictions.append(pred_num)
            step_record["parsed"] = fc_parsed
            display_pred = raw_pred if weather else pred_num
            print(f"  step {t:2d}: pred={display_pred}  true={true_val}",
                  flush=(t % 5 == 4))
        except Exception as e:
            print(f"  step {t:2d}: PARSE ERROR: {e}")
            parse_errors.append((t, "forecast", str(e), fc_content))
            predictions.append(0)
            step_record["parse_error"] = str(e)

        _save_step(step_record)

        # ── feedback turn (model produces note) ─────────────────────────────
        t_fb_start = time.time()
        feedback_input = json.dumps({
            "turn_type": "feedback",
            "note": note,
            **ep_dict,
            label_key: true_val,
        })
        messages.append({"role": "user", "content": feedback_input})
        _save_conv({"role": "user", "content": feedback_input})

        try:
            fb_content, fb_reasoning, fb_usage = call_api(messages, model_key)
        except Exception as e:
            _save_step({"split": split, "idx": idx, "model_key": model_key,
                        "prompt": prompt_name, "step": t, "turn": "feedback",
                        "api_error": str(e), "elapsed_s": round(time.time() - t_fb_start, 2)})
            print(f"  step {t:2d}: FEEDBACK API ERROR: {e}", flush=True)
            raise

        _accumulate(tokens, fb_usage)
        step_usage_log[-1]["feedback"] = fb_usage
        messages.append({"role": "assistant", "content": fb_content})
        _save_conv({"role": "assistant", "content": fb_content})

        fb_record = {"split": split, "idx": idx, "model_key": model_key,
                     "prompt": prompt_name, "step": t, "turn": "feedback",
                     "note_in": note, "raw_response": fb_content, "usage": fb_usage,
                     "elapsed_s": round(time.time() - t_fb_start, 2)}
        if fb_reasoning:
            fb_record["reasoning"] = fb_reasoning

        try:
            fb_parsed = parse_model_response(fb_content)
            new_note = fb_parsed.get("note", note)
            note = new_note if isinstance(new_note, str) else str(new_note)
            notes.append(note)
            fb_record["parsed"] = fb_parsed
            fb_record["note_out"] = note
        except Exception as e:
            print(f"  step {t:2d}: FEEDBACK PARSE ERROR: {e}")
            parse_errors.append((t, "feedback", str(e), fb_content))
            fb_record["parse_error"] = str(e)

        _save_step(fb_record)
        _save_usage({"step": t, "usage": {"forecast": fc_usage, "feedback": fb_usage},
                     "tokens_cumulative": dict(tokens)})

        # timing
        if (t + 1) % 10 == 0:
            elapsed_chunk = time.time() - t_chunk_start
            elapsed_total = time.time() - t_instance_start
            msg = (f"  [timer] steps {t-8:>3d}-{t:>3d}: {elapsed_chunk:.1f}s  "
                   f"total: {elapsed_total:.1f}s  ({t+1}/{T})")
            print(msg, flush=True)
            _log_timing(f"{instance_tag}  steps {t-8:>3d}-{t:>3d}: {elapsed_chunk:.1f}s  total: {elapsed_total:.1f}s  ({t+1}/{T})")
            t_chunk_start = time.time()

    # ── save final results ───────────────────────────────────────────────────
    elapsed = time.time() - t_instance_start
    cost = _estimate_cost(tokens, model_key)
    analysis = analyze_online_output(case, predictions)
    ground_truth = get_ground_truth_outputs(case)

    result = {
        "case_id":  f"{split}_{idx:03d}",
        "meta": {
            "split":     split,
            "idx":       idx,
            "run":       run_dir.name,
            "mode":      "note_with_history",
            "model":     MODEL_MAP[model_key],
            "model_key": model_key,
            "prompt":    prompt_name,
            "n_steps":   T,
            "elapsed_s": round(elapsed, 1),
        },
        "token_usage": tokens,
        "estimated_cost": cost,
        "analysis":     analysis,
        "input": {
            "n":                  len(case["expert_predictions"][0]),
            "T":                  T,
            "eta":                case["learning_rate"],
            "expert_predictions": case["expert_predictions"],
            "true_labels":        case["true_labels"],
            "losses":             case["losses"],
        },
        "ground_truth": ground_truth,
        "response":     {"predictions": predictions, "notes": notes},
    }
    if parse_errors:
        result["meta"]["parse_errors"] = parse_errors

    (run_dir / "result.json").write_text(json.dumps(result, indent=2))
    (run_dir / "token_usage.json").write_text(json.dumps(
        {"total": tokens, "cost": cost, "per_step": step_usage_log}, indent=2))

    plot_online_regret(analysis, save_path=run_dir / "plot_regret.png")
    plot_online_loss(analysis, save_path=run_dir / "plot_loss.png")

    print(f"\nSaved to: {run_dir}/")
    print(f"  accuracy: {analysis['accuracy']:.3f}  regret: {analysis['regret_curve'][-1]}")
    if parse_errors:
        print(f"  Warning: {len(parse_errors)} parse error(s)")

    msg = f"DONE {instance_tag}  total: {elapsed:.1f}s  ({elapsed/T:.2f}s/step)"
    cost_msg = (f"  tokens: {tokens['prompt_tokens']:,} in + {tokens['completion_tokens']:,} out  cost: ${cost['total']:.4f}")
    print(f"\n  [timer] {msg}", flush=True)
    print(cost_msg, flush=True)
    _log_timing(msg)
    _log_timing(cost_msg)

    _update_regret_table(model_key, prompt_name, split, idx, analysis["regret_curve"])
    return run_dir


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_multi(args, flag):
    """Collect all values after *flag* until the next ``--`` flag or end."""
    if flag not in args:
        return []
    start = args.index(flag) + 1
    vals = []
    for i in range(start, len(args)):
        if args[i].startswith("--"):
            break
        vals.append(args[i])
    return vals


def main():
    global DATASET_PATH

    args = sys.argv[1:]
    if len(args) < 2 or "--model" not in args:
        print(__doc__)
        sys.exit(1)

    split = args[0]

    # --dataset (optional, default DATASET_PATH)
    if "--dataset" in args:
        DATASET_PATH = args[args.index("--dataset") + 1]

    # --idx: one or more indices, or "all"
    idx_args = _parse_multi(args, "--idx")
    if not idx_args:
        # backward compat: positional idx right after split
        try:
            idx_args = [args[1]]
        except IndexError:
            print("ERROR: provide --idx <i ...> or --idx all")
            sys.exit(1)

    data = load_data(DATASET_PATH)
    if idx_args == ["all"]:
        indices = list(range(len(data[split])))
    else:
        indices = [int(x) for x in idx_args]

    models       = _parse_multi(args, "--model")
    prompts      = _parse_multi(args, "--prompt") or [DEFAULT_PROMPT]
    plot_weights = "--plot-weights" in args
    max_steps    = int(args[args.index("--steps") + 1]) if "--steps" in args else None

    for mk in models:
        if mk not in MODEL_MAP:
            print(f"ERROR: unknown model '{mk}', must be one of {list(MODEL_MAP)}")
            sys.exit(1)

    total = len(indices) * len(models) * len(prompts)
    done  = 0
    for mk in models:
        for pn in prompts:
            for idx in indices:
                done += 1
                print(f"\n[{done}/{total}] model={mk}  prompt={pn}  idx={idx}")
                print("-" * 60)
                if pn in NOTE_HISTORY_PROMPTS:
                    run_note_with_history(split, idx, mk, pn, max_steps=max_steps)
                elif pn in TWO_TURN_PROMPTS:
                    run_two_turn(split, idx, mk, pn, max_steps=max_steps)
                elif pn in NONOTE_PROMPTS:
                    run_nonote(split, idx, mk, pn, max_steps=max_steps)
                elif pn in ONLINE_PROMPTS:
                    run_interactive_online_once(split, idx, mk, pn, max_steps=max_steps)
                else:
                    run_interactive_once(split, idx, mk, pn, plot_weights=plot_weights, max_steps=max_steps)

    print(f"\nAll done. {total} combos processed.")


if __name__ == "__main__":
    main()
