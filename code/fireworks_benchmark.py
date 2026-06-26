import csv
import json
import os
import time
from datetime import datetime

import requests

from config import (
    FIREWORKS_API_KEY,
    FIREWORKS_BASE_URL,
    FIREWORKS_MODEL_NAME,
    NUM_RUNS,
    REQUEST_TIMEOUT,
    MAX_TOKENS,
    TEMPERATURE,
)


PLATFORM_NAME = "Fireworks"


def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_prompts():
    prompts_path = os.path.join(os.path.dirname(__file__), "prompts.json")

    with open(prompts_path, "r", encoding="utf-8") as file:
        return json.load(file)


def call_fireworks(prompt_text):
    headers = {
        "Authorization": f"Bearer {FIREWORKS_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": FIREWORKS_MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant. Keep the response concise.",
            },
            {
                "role": "user",
                "content": prompt_text,
            },
        ],
        "stream": False,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "perf_metrics_in_response": True,
    }

    start_time = time.perf_counter()

    try:
        response = requests.post(
            FIREWORKS_BASE_URL,
            headers=headers,
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )

        latency = time.perf_counter() - start_time

        try:
            response_json = response.json()
        except Exception:
            response_json = {}

        return {
            "success": response.status_code == 200,
            "status_code": response.status_code,
            "latency_sec": round(latency, 4),
            "response_json": response_json,
            "error_message": "" if response.status_code == 200 else response.text[:500],
        }

    except requests.exceptions.Timeout:
        latency = time.perf_counter() - start_time

        return {
            "success": False,
            "status_code": "timeout",
            "latency_sec": round(latency, 4),
            "response_json": {},
            "error_message": "Request timed out",
        }

    except Exception as error:
        latency = time.perf_counter() - start_time

        return {
            "success": False,
            "status_code": "error",
            "latency_sec": round(latency, 4),
            "response_json": {},
            "error_message": str(error),
        }


def extract_usage(response_json):
    usage = response_json.get("usage", {})

    input_tokens = usage.get("prompt_tokens", "")
    output_tokens = usage.get("completion_tokens", "")
    total_tokens = usage.get("total_tokens", "")

    return input_tokens, output_tokens, total_tokens


def extract_perf_metrics(response_json):
    metrics = response_json.get("perf_metrics", {})

    server_ttft = metrics.get("server-time-to-first-token", "")
    server_processing_time = metrics.get("server-processing-time", "")

    return server_ttft, server_processing_time


def extract_response_chars(response_json):
    try:
        content = response_json["choices"][0]["message"]["content"]
        return len(content)
    except Exception:
        return 0


def calculate_throughput(output_tokens, latency_sec):
    try:
        if output_tokens == "" or latency_sec <= 0:
            return ""

        return round(float(output_tokens) / float(latency_sec), 4)
    except Exception:
        return ""


def percentile(values, p):
    if not values:
        return ""

    values = sorted(values)
    index = round((p / 100) * (len(values) - 1))
    return values[index]


def save_csv(rows, file_path):
    if not rows:
        return

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def create_summary(rows):
    successful_rows = [row for row in rows if row["success"] is True]
    successful_latencies = [row["latency_sec"] for row in successful_rows]

    throughput_values = [
        row["throughput_tokens_per_sec"]
        for row in successful_rows
        if row["throughput_tokens_per_sec"] != ""
    ]

    total_requests = len(rows)
    successful_requests = len(successful_rows)
    failed_requests = total_requests - successful_requests

    avg_latency = (
        round(sum(successful_latencies) / len(successful_latencies), 4)
        if successful_latencies
        else ""
    )

    avg_throughput = (
        round(sum(throughput_values) / len(throughput_values), 4)
        if throughput_values
        else ""
    )

    return [
        {
            "platform": PLATFORM_NAME,
            "model": FIREWORKS_MODEL_NAME,
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "success_rate_percent": round((successful_requests / total_requests) * 100, 2)
            if total_requests
            else 0,
            "avg_latency_sec_successful_only": avg_latency,
            "p50_latency_sec_successful_only": percentile(successful_latencies, 50),
            "p95_latency_sec_successful_only": percentile(successful_latencies, 95),
            "avg_throughput_tokens_per_sec": avg_throughput,
        }
    ]


def run_benchmark():
    prompts = load_prompts()
    results = []

    print(f"Running {PLATFORM_NAME} benchmark")
    print(f"Prompts: {len(prompts)}")
    print(f"Runs per prompt: {NUM_RUNS}")
    print(f"Total requests: {len(prompts) * NUM_RUNS}")

    for run_number in range(1, NUM_RUNS + 1):
        for prompt in prompts:
            result = call_fireworks(prompt["prompt"])

            input_tokens, output_tokens, total_tokens = extract_usage(
                result["response_json"]
            )

            server_ttft, server_processing_time = extract_perf_metrics(
                result["response_json"]
            )

            throughput = calculate_throughput(
                output_tokens=output_tokens,
                latency_sec=result["latency_sec"],
            )

            row = {
                "platform": PLATFORM_NAME,
                "model": FIREWORKS_MODEL_NAME,
                "prompt_id": prompt["id"],
                "category": prompt["category"],
                "run_number": run_number,
                "success": result["success"],
                "status_code": result["status_code"],
                "latency_sec": result["latency_sec"],
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "throughput_tokens_per_sec": throughput,
                "server_ttft_sec": server_ttft,
                "server_processing_time_sec": server_processing_time,
                "response_chars": extract_response_chars(result["response_json"]),
                "timestamp": datetime.now().isoformat(),
                "error_message": result["error_message"],
            }

            results.append(row)

            print(
                f"Done: Run {run_number}, Prompt {prompt['id']}, "
                f"Success: {result['success']}, Latency: {result['latency_sec']}s"
            )

    project_root = get_project_root()

    raw_file = os.path.join(project_root, "results", "fireworks_results.csv")
    summary_file = os.path.join(project_root, "results", "fireworks_summary.csv")

    save_csv(results, raw_file)
    save_csv(create_summary(results), summary_file)

    print("\nBenchmark complete")
    print(f"Raw results saved to: {raw_file}")
    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    run_benchmark()