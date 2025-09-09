#!/usr/bin/env python3
"""
Multi-turn latency harness

Produces:
 - latency_metrics.json : aggregated metrics per-config, per-turn, overall (milliseconds)
 - samples.jsonl        : raw per-request samples with assistant_text (one JSON per line)

Usage:
  python latency_harness.py

Notes:
 - Assumes OpenAI Azure SDK usage via openai.AzureOpenAI(...), streaming via client.chat.completions.create(stream=True,...)
 - If your SDK is async, replace run_in_executor usage with direct awaits.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

from openai.types.chat import ChatCompletionFunctionToolParam
from progress.bar import ChargingBar


# ---------- Utilities ----------
def percentile(sorted_samples: List[float], p: float) -> float:
    """
    Compute p-th percentile (0 <= p <= 100) via linear interpolation.
    `sorted_samples` must be pre-sorted ascending.
    """
    if not sorted_samples:
        raise ValueError("Empty sample list for percentile")
    if not (0.0 <= p <= 100.0):
        raise ValueError("p must be between 0 and 100")

    n = len(sorted_samples)
    if n == 1:
        return float(sorted_samples[0])

    pos = (p / 100.0) * (n - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(sorted_samples[int(pos)])
    w = pos - lo
    return float(sorted_samples[lo] * (1.0 - w) + sorted_samples[hi] * w)


# ---------- Blocking streaming call (executed in threadpool) ----------
def call_openai_blocking_messages(
    api_key: str,
    api_endpoint: str,
    api_version: str,
    deployment: str,
    messages: List[Dict[str, str]],
    tools: List[ChatCompletionFunctionToolParam],
) -> Tuple[Optional[float], Optional[str]]:
    """
    Execute a blocking streaming chat completion. Returns (time_to_first_chunk_seconds_or_None, assistant_text_or_None).
    Parses streaming chunks heuristically for various SDK formats. Defensive: returns (None,None) on error.
    """
    try:
        import openai  # local import (safer for threadpool worker)
    except Exception as exc:
        print(f"[call_openai_blocking_messages] error importing openai: {exc}")
        return (None, None)

    try:
        client = openai.AzureOpenAI(
            api_version=api_version,
            api_key=api_key,
            azure_endpoint=api_endpoint,
            azure_deployment=deployment,
        )
    except Exception as exc:
        print(f"[call_openai_blocking_messages] error creating AzureOpenAI client: {exc}")
        return (None, None)

    start_ts = time.time()
    first_chunk_ts: Optional[float] = None
    assistant_parts: List[str] = []

    try:
        stream = client.chat.completions.create(
            stream=True,
            model=deployment,
            messages=messages,
            tools=tools,
        )
    except Exception as exc:
        print(f"[call_openai_blocking_messages] create() failed: {exc}")
        return None, None

    # iterate the streaming iterator fully (we extract first-chunk time and accumulate text)
    try:
        for chunk in stream:
            # record TTF at first observed chunk
            if first_chunk_ts is None:
                first_chunk_ts = time.time()

            # Parse chunk defensively: many SDKs return dict-like chunks with choices -> delta/message/text
            try:
                if isinstance(chunk, dict):
                    choices = chunk.get("choices") or []
                    if choices and isinstance(choices, list):
                        c0 = choices[0]
                        # 1) streaming delta
                        delta = c0.get("delta")
                        if isinstance(delta, dict):
                            # delta might be nested { "content": "..." } or { "content": {"parts": [...] } }
                            content = delta.get("content")
                            if isinstance(content, str):
                                assistant_parts.append(content)
                            elif isinstance(content, dict):
                                # some formats embed parts
                                parts = content.get("parts")
                                if isinstance(parts, list):
                                    assistant_parts.extend([str(p) for p in parts if p is not None])
                        # 2) full message
                        msg = c0.get("message")
                        if isinstance(msg, dict):
                            content = msg.get("content")
                            if isinstance(content, str):
                                assistant_parts.append(content)
                            elif isinstance(content, dict):
                                # message.content may have "parts"
                                parts = content.get("parts")
                                if isinstance(parts, list):
                                    assistant_parts.extend([str(p) for p in parts if p is not None])
                        # 3) fallback: text field
                        text_fallback = chunk.get("text")
                        if isinstance(text_fallback, str):
                            assistant_parts.append(text_fallback)
                    else:
                        # chunk may be other dict shape; try to extract text or data
                        for key in ("text", "data"):
                            v = chunk.get(key)
                            if isinstance(v, str):
                                assistant_parts.append(v)
                else:
                    # non-dict chunk; coerce to string
                    assistant_parts.append(str(chunk))
            except Exception:
                # parsing errors should not abort the iteration
                continue

    except Exception as exc:
        # streaming iteration failed unexpectedly
        print(f"[call_openai_blocking_messages] streaming iteration error: {exc}")
        # if we already observed first chunk, use that time; otherwise consider failed
        if first_chunk_ts is None:
            return None, None

    # compute durations / result
    if first_chunk_ts is None:
        first_chunk_ts = time.time()

    assistant_text = "".join(assistant_parts).strip()
    if assistant_text == "":
        assistant_text = None

    return (first_chunk_ts - start_ts, assistant_text)


# ---------- Worker: one iteration (sequential turns) ----------
async def iteration_worker(
    config_name: str,
    iteration_id: int,
    sem: asyncio.Semaphore,
    loop: asyncio.AbstractEventLoop,
    out_queue: asyncio.Queue,
    api_key: str,
    api_endpoint: str,
    api_version: str,
    deployment: str,
    instructions: str,
    simulation: List[str],
):
    """
    Perform one iteration. For each turn:
      - build messages = [system] + prior_history + [current user]
      - run blocking streaming call inside semaphore (in executor)
      - append assistant response to prior_history so the next turn includes it
      - push a per-sample record to out_queue:
          { "config": config_name, "iteration": iteration_id, "turn_index": turn_idx, "duration_s": float|None, "assistant_text": str|None }
    """
    prior_history: List[Dict[str, str]] = []  # includes both user and assistant dict messages
    for turn_index, user_turn in enumerate(simulation):
        messages = [{"role": "system", "content": instructions}] + prior_history + [{"role": "user", "content": user_turn}]

        # run the blocking call: hold semaphore for the duration of the request
        async with sem:
            fn = partial(call_openai_blocking_messages, api_key, api_endpoint, api_version, deployment, messages)
            duration_and_text = await loop.run_in_executor(None, fn)

        # duration_and_text is (duration_s_or_None, assistant_text_or_None)
        duration_s, assistant_text = duration_and_text

        # Always append user message; append assistant message even if empty to keep turn order
        prior_history.append({"role": "user", "content": user_turn})
        prior_history.append({"role": "assistant", "content": assistant_text or ""})

        # create sample record and push to queue
        sample = {
            "config": config_name,
            "iteration": iteration_id,
            "turn_index": turn_index,
            "duration_s": duration_s,
            "assistant_text": assistant_text,
            "timestamp": time.time(),
        }
        await out_queue.put(sample)


# ---------- Main runner ----------
async def run_and_compute_metrics(
    config_path: str = "config.json",
    concurrency_per_config: int = 1,
    out_metrics_path: str = "latency_metrics.json",
    out_samples_path: str = "samples.jsonl",
):
    with open(config_path, "r") as fh:
        configs = json.load(fh)

    loop = asyncio.get_running_loop()

    # aggregator structures
    overall_samples: List[float] = []
    results: Dict[str, Any] = {}

    # We'll write samples incrementally to a .jsonl file while consuming from queue
    samples_fh = open(out_samples_path, "w", encoding="utf-8")

    try:
        for config in configs:
            name = config.get("name", "unnamed")
            count = int(config.get("count", 1))
            llm = config.get("llm", {})
            api_version = llm.get("azure_api_version", "")
            api_key = llm.get("azure_api_key", "")
            api_endpoint = llm.get("azure_endpoint", "")
            deployment = llm.get("azure_deployment", "")

            instructions = config.get("instructions", "")
            simulation = config.get("simulation", [])
            if not simulation:
                print(f"[warning] config {name} has empty simulation; skipping")
                continue

            num_turns = len(simulation)
            total_requests = count * num_turns
            print(f"\n=== Config '{name}': iterations={count}, turns={num_turns}, total_requests={total_requests} ===")
            print(f"  endpoint={api_endpoint} deployment={deployment}")

            # per-config storage
            per_turn_samples: Dict[int, List[float]] = {i: [] for i in range(num_turns)}
            per_turn_failures: Dict[int, int] = {i: 0 for i in range(num_turns)}

            # concurrency control and queue
            sem = asyncio.Semaphore(concurrency_per_config)
            queue: asyncio.Queue = asyncio.Queue()

            # spawn iteration workers (each worker has its own iteration_id)
            workers = [
                asyncio.create_task(
                    iteration_worker(
                        config_name=name,
                        iteration_id=i,
                        sem=sem,
                        loop=loop,
                        out_queue=queue,
                        api_key=api_key,
                        api_endpoint=api_endpoint,
                        api_version=api_version,
                        deployment=deployment,
                        instructions=instructions,
                        simulation=simulation,
                    )
                )
                for i in range(count)
            ]

            # progress bar counts requests (turns)
            bar = ChargingBar(name, max=total_requests)

            # consume queue until we've seen all expected samples
            seen = 0
            while seen < total_requests:
                sample = await queue.get()
                seen += 1

                # write raw sample to jsonl
                samples_fh.write(json.dumps(sample, ensure_ascii=False) + "\n")
                samples_fh.flush()

                # update per-turn aggregates
                turn_idx = sample["turn_index"]
                duration_s = sample["duration_s"]
                if duration_s is None:
                    per_turn_failures[turn_idx] += 1
                else:
                    per_turn_samples[turn_idx].append(duration_s)
                    overall_samples.append(duration_s)
                bar.next()

            bar.finish()

            # ensure all iteration workers completed
            await asyncio.gather(*workers, return_exceptions=True)

            # compute per-turn metrics
            per_turn_metrics: Dict[str, Any] = {}
            for turn_idx in sorted(per_turn_samples.keys()):
                samples = per_turn_samples[turn_idx]
                metrics: Dict[str, Optional[float]] = {}
                if samples:
                    s_sorted = sorted(samples)
                    metrics["count_samples"] = len(s_sorted)
                    metrics["p50_ms"] = percentile(s_sorted, 50) * 1000.0
                    metrics["p95_ms"] = percentile(s_sorted, 95) * 1000.0
                    metrics["min_ms"] = float(s_sorted[0]) * 1000.0
                    metrics["max_ms"] = float(s_sorted[-1]) * 1000.0
                    metrics["mean_ms"] = float(sum(s_sorted) / len(s_sorted)) * 1000.0
                else:
                    metrics["count_samples"] = 0
                    metrics["p50_ms"] = None
                    metrics["p95_ms"] = None
                    metrics["min_ms"] = None
                    metrics["max_ms"] = None
                    metrics["mean_ms"] = None
                metrics["failures"] = per_turn_failures[turn_idx]
                per_turn_metrics[f"turn_{turn_idx + 1}"] = metrics

            # config aggregated metrics (across all turns)
            aggregated_samples = [v for lst in per_turn_samples.values() for v in lst]
            agg_metrics: Dict[str, Optional[float]] = {}
            if aggregated_samples:
                s_sorted = sorted(aggregated_samples)
                agg_metrics["count_samples"] = len(s_sorted)
                agg_metrics["p50_ms"] = percentile(s_sorted, 50) * 1000.0
                agg_metrics["p95_ms"] = percentile(s_sorted, 95) * 1000.0
                agg_metrics["min_ms"] = float(s_sorted[0]) * 1000.0
                agg_metrics["max_ms"] = float(s_sorted[-1]) * 1000.0
                agg_metrics["mean_ms"] = float(sum(s_sorted) / len(s_sorted)) * 1000.0
            else:
                agg_metrics["count_samples"] = 0
                agg_metrics["p50_ms"] = None
                agg_metrics["p95_ms"] = None
                agg_metrics["min_ms"] = None
                agg_metrics["max_ms"] = None
                agg_metrics["mean_ms"] = None
            agg_metrics["total_failures"] = sum(per_turn_failures.values())

            results[name] = {
                "config": config,
                "per_turn_metrics": per_turn_metrics,
                "aggregated_metrics": agg_metrics,
            }

            # print concise summary
            print(f"\nSummary for config '{name}':")
            print(f"  Aggregated samples={agg_metrics['count_samples']} failures={agg_metrics['total_failures']}")
            if agg_metrics["count_samples"]:
                print(f"  p50={agg_metrics['p50_ms']:.2f} ms  p95={agg_metrics['p95_ms']:.2f} ms  mean={agg_metrics['mean_ms']:.2f} ms")
            else:
                print("  No successful aggregated samples.")

            print("  Per-turn summary:")
            for tname, m in per_turn_metrics.items():
                if m["count_samples"]:
                    print(f"    {tname}: samples={m['count_samples']} failures={m['failures']} p50={m['p50_ms']:.2f} ms p95={m['p95_ms']:.2f} ms")
                else:
                    print(f"    {tname}: samples=0 failures={m['failures']}")

        # overall across all configs
        overall_summary: Dict[str, Optional[float]] = {}
        if overall_samples:
            s_sorted = sorted(overall_samples)
            overall_summary["total_samples"] = len(s_sorted)
            overall_summary["p50_ms"] = percentile(s_sorted, 50) * 1000.0
            overall_summary["p95_ms"] = percentile(s_sorted, 95) * 1000.0
            overall_summary["min_ms"] = float(s_sorted[0]) * 1000.0
            overall_summary["max_ms"] = float(s_sorted[-1]) * 1000.0
            overall_summary["mean_ms"] = float(sum(s_sorted) / len(s_sorted)) * 1000.0
            print("\n=== Overall across all configs ===")
            print(f"  total_samples={overall_summary['total_samples']} p50={overall_summary['p50_ms']:.2f} ms p95={overall_summary['p95_ms']:.2f} ms")
        else:
            overall_summary["total_samples"] = 0
            print("\n=== Overall: no successful samples collected ===")

        # write metrics artifact
        metrics_output = {
            "per_config": {},
            "overall": overall_summary,
        }
        for cfg_name, payload in results.items():
            metrics_output["per_config"][cfg_name] = {
                "per_turn": payload["per_turn_metrics"],
                "aggregated": payload["aggregated_metrics"],
            }

        with open(out_metrics_path, "w", encoding="utf-8") as outf:
            json.dump(metrics_output, outf, indent=2, ensure_ascii=False)

        print(f"\nSaved aggregated metrics to {out_metrics_path}")
        print(f"Saved raw samples (incremental) to {out_samples_path}")

    finally:
        samples_fh.close()


# ---------- CLI ----------
if __name__ == "__main__":
    # Tune concurrency_per_config according to your environment and API rate limits.
    asyncio.run(run_and_compute_metrics(config_path="config.json", concurrency_per_config=3))
