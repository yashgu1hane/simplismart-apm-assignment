import os
import csv
import json
import time
import requests
from datetime import datetime, UTC

from config import PLATFORM_NAME, API_KEY, MODEL_NAME, BASE_URL, NUM_RUNS, REQUEST_TIMEOUT


def utc_now():
    return datetime.now(UTC).isoformat()


def load_prompts():
    current_dir = os.path.dirname(__file__)
    prompts_path = os.path.join(current_dir, "prompts.json")

    with open(prompts_path, "r", encoding="utf-8") as file:
        return json.load(file)


def get_results_dir():
    project_root = os.path.dirname(os.path.dirname(__file__))
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def percentile(values, percentile_value):
    if not values:
        return None

    sorted_values = sorted(values)
    index = int(round((percentile_value / 100) * (len(sorted_values) - 1)))
    return sorted_values[index]


def safe_round(value, digits=4):
    if value is None or value == "":
        return ""

    try:
        return round(float(value), digits)
    except Exception:
        return ""


def call_llm(prompt_text):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant. Keep answers concise and follow the user's constraints."
            },
            {
                "role": "user",
                "content": prompt_text
            }
        ],
        "stream": False,
        "temperature": 0.2,
        "max_tokens": 100,
        "perf_metrics_in_response": True
    }

    # Fireworks-specific flag to get provider-side metrics
    #if PLATFORM_NAME.lower() == "fireworks":
        #payload["perf_metrics_in_response"] = True

    start_time = time.perf_counter()

    response = requests.post(
        BASE_URL,
        headers=headers,
        json=payload,
        timeout=REQUEST_TIMEOUT
    )

    end_time = time.perf_counter()

    local_latency_sec = end_time - start_time

    return response, local_latency_sec


def extract_fireworks_perf_metrics(response_json):
    """
    Fireworks returns provider-side metrics when:
    perf_metrics_in_response = True

    These metrics are useful but should be treated as secondary
    because Simplismart dedicated endpoint does not expose equivalent metrics.
    """

    perf_metrics = response_json.get("perf_metrics", {})

    server_processing_time_sec = perf_metrics.get("server-processing-time")
    server_ttft_sec = perf_metrics.get("server-time-to-first-token")

    return server_processing_time_sec, server_ttft_sec


def run_benchmark():
    prompts = load_prompts()
    results = []

    print(f"Starting benchmark for: {PLATFORM_NAME}")
    print(f"Model: {MODEL_NAME}")
    print(f"Total prompts: {len(prompts)}")
    print(f"Runs per prompt: {NUM_RUNS}")
    print("-" * 60)

    for run_number in range(1, NUM_RUNS + 1):
        for prompt in prompts:
            prompt_id = prompt["id"]
            category = prompt["category"]
            prompt_text = prompt["prompt"]

            try:
                response, local_latency_sec = call_llm(prompt_text)
                status_code = response.status_code

                try:
                    response_json = response.json()
                except Exception:
                    response_json = {}

                success = status_code == 200

                usage = response_json.get("usage", {})
                input_tokens = usage.get("prompt_tokens")
                output_tokens = usage.get("completion_tokens")
                total_tokens = usage.get("total_tokens")

                request_id = response_json.get("id", "")

                server_processing_time_sec = ""
                server_ttft_sec = ""

                if PLATFORM_NAME.lower() == "fireworks":
                    server_processing_time_sec, server_ttft_sec = extract_fireworks_perf_metrics(response_json)

                response_chars = 0
                response_preview = ""
                error_message = ""

                if success:
                    choices = response_json.get("choices", [])
                    if choices:
                        content = choices[0].get("message", {}).get("content", "")
                        response_chars = len(content)
                        response_preview = content[:200].replace("\n", " ")
                else:
                    error_message = response.text[:300]

                throughput = None
                if output_tokens is not None and local_latency_sec > 0:
                    throughput = output_tokens / local_latency_sec

                result_row = {
                    "platform": PLATFORM_NAME,
                    "model": MODEL_NAME,
                    "prompt_id": prompt_id,
                    "category": category,
                    "run_number": run_number,
                    "success": success,
                    "http_status": status_code,

                    # Primary API-level metrics
                    "api_latency_sec": safe_round(local_latency_sec),
                    "input_tokens": input_tokens if input_tokens is not None else "",
                    "output_tokens": output_tokens if output_tokens is not None else "",
                    "total_tokens": total_tokens if total_tokens is not None else "",
                    "throughput_tokens_per_sec": safe_round(throughput),

                    # Secondary provider/system metrics, available mainly for Fireworks
                    "server_processing_time_sec": safe_round(server_processing_time_sec),
                    "server_ttft_sec": safe_round(server_ttft_sec),

                    # Other metadata
                    "response_chars": response_chars,
                    "response_preview": response_preview,
                    "request_id": request_id,
                    "timestamp": utc_now(),
                    "error_message": error_message
                }

                results.append(result_row)

                print(
                    f"Done: Run {run_number}, Prompt {prompt_id}, "
                    f"Status {status_code}, API Latency {safe_round(local_latency_sec)} sec"
                )

            except Exception as e:
                result_row = {
                    "platform": PLATFORM_NAME,
                    "model": MODEL_NAME,
                    "prompt_id": prompt_id,
                    "category": category,
                    "run_number": run_number,
                    "success": False,
                    "http_status": "",

                    # Primary API-level metrics
                    "api_latency_sec": "",
                    "input_tokens": "",
                    "output_tokens": "",
                    "total_tokens": "",
                    "throughput_tokens_per_sec": "",

                    # Secondary provider/system metrics
                    "server_processing_time_sec": "",
                    "server_ttft_sec": "",

                    # Other metadata
                    "response_chars": "",
                    "response_preview": "",
                    "request_id": "",
                    "timestamp": utc_now(),
                    "error_message": str(e)
                }

                results.append(result_row)

                print(f"Failed: Run {run_number}, Prompt {prompt_id} - {e}")

    save_results(results)
    save_summary(results)

    print("-" * 60)
    print("Benchmark complete")


def save_results(results):
    results_dir = get_results_dir()
    filename = os.path.join(results_dir, f"{PLATFORM_NAME}_results.csv")

    with open(filename, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"Raw results saved to: {filename}")


def save_summary(results):
    results_dir = get_results_dir()
    filename = os.path.join(results_dir, f"{PLATFORM_NAME}_summary.csv")

    successful_results = [row for row in results if row["success"] is True]

    total_requests = len(results)
    successful_requests = len(successful_results)
    failed_requests = total_requests - successful_requests

    success_rate = (successful_requests / total_requests) * 100 if total_requests > 0 else 0

    api_latencies = [
        float(row["api_latency_sec"])
        for row in successful_results
        if row["api_latency_sec"] != ""
    ]

    input_tokens = [
        int(row["input_tokens"])
        for row in successful_results
        if row["input_tokens"] != ""
    ]

    output_tokens = [
        int(row["output_tokens"])
        for row in successful_results
        if row["output_tokens"] != ""
    ]

    throughputs = [
        float(row["throughput_tokens_per_sec"])
        for row in successful_results
        if row["throughput_tokens_per_sec"] != ""
    ]

    server_processing_times = [
        float(row["server_processing_time_sec"])
        for row in successful_results
        if row["server_processing_time_sec"] != ""
    ]

    server_ttfts = [
        float(row["server_ttft_sec"])
        for row in successful_results
        if row["server_ttft_sec"] != ""
    ]

    summary = {
        "platform": PLATFORM_NAME,
        "model": MODEL_NAME,
        "total_requests": total_requests,
        "successful_requests": successful_requests,
        "failed_requests": failed_requests,
        "success_rate_percent": safe_round(success_rate, 2),

        # Primary API-level summary
        "avg_api_latency_sec": safe_round(sum(api_latencies) / len(api_latencies)) if api_latencies else "",
        "p50_api_latency_sec": safe_round(percentile(api_latencies, 50)) if api_latencies else "",
        "p95_api_latency_sec": safe_round(percentile(api_latencies, 95)) if api_latencies else "",
        "cold_start_api_latency_sec": safe_round(api_latencies[0]) if api_latencies else "",
        "avg_input_tokens": safe_round(sum(input_tokens) / len(input_tokens), 2) if input_tokens else "",
        "avg_output_tokens": safe_round(sum(output_tokens) / len(output_tokens), 2) if output_tokens else "",
        "avg_throughput_tokens_per_sec": safe_round(sum(throughputs) / len(throughputs)) if throughputs else "",

        # Secondary provider/system summary
        "avg_server_processing_time_sec": safe_round(sum(server_processing_times) / len(server_processing_times)) if server_processing_times else "",
        "avg_server_ttft_sec": safe_round(sum(server_ttfts) / len(server_ttfts)) if server_ttfts else "",

        "benchmark_timestamp": utc_now()
    }

    with open(filename, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=summary.keys())
        writer.writeheader()
        writer.writerow(summary)

    print(f"Summary saved to: {filename}")


if __name__ == "__main__":
    run_benchmark()