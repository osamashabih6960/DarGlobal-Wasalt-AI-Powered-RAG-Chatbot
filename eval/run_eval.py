
"""Evaluate the DarGlobal x Wasalt RAG chatbot using MLflow."""

import json
import os
import re
import sys
import time
import unicodedata
import uuid
from pathlib import Path
from statistics import mean

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

import mlflow
from app.graph import TOP_K, ask, build_graph


# --------------------------------------------------
# Configuration
# --------------------------------------------------

EVAL_FILE = PROJECT_ROOT / "eval" / "qa.json"
EXPERIMENT_NAME = "dar-wasalt-chatbot-eval"
RESULTS_FILE = PROJECT_ROOT / "eval" / "eval_results.json"

REQUEST_DELAY_SECONDS = 8
MAX_RETRIES = 3


# --------------------------------------------------
# MLflow configuration
# --------------------------------------------------

def configure_mlflow():
    """Configure MLflow experiment tracking."""

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(EXPERIMENT_NAME)


# --------------------------------------------------
# Evaluation dataset
# --------------------------------------------------

def load_test_cases():
    """Load evaluation questions from JSON."""

    if not EVAL_FILE.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {EVAL_FILE}"
        )

    with EVAL_FILE.open("r", encoding="utf-8") as file:
        cases = json.load(file)

    if not isinstance(cases, list) or not cases:
        raise ValueError(
            "Evaluation dataset must be a non-empty JSON list."
        )

    for index, case in enumerate(cases, start=1):
        if not isinstance(case, dict):
            raise ValueError(
                f"Invalid evaluation case at position {index}"
            )

        if not case.get("question") or not isinstance(
            case.get("keywords"), list
        ):
            raise ValueError(
                f"Invalid question or keywords at position {index}"
            )

    return cases


# --------------------------------------------------
# Text normalization
# --------------------------------------------------

def normalize_text(text):
    """Normalize text for reliable keyword matching."""

    text = unicodedata.normalize("NFKC", str(text))
    text = text.lower()

    # Normalize Unicode hyphens and dashes.
    text = re.sub(r"[\u2010-\u2015\u2212]", "-", text)

    # Normalize special spaces and repeated whitespace.
    text = re.sub(r"\s+", " ", text)

    # Normalize percentage formatting: 66 % -> 66%
    text = re.sub(r"(?<=\d)\s+(?=%)", "", text)

    return text.strip()


# --------------------------------------------------
# Answer evaluation
# --------------------------------------------------

def evaluate_answer(answer, expected_keywords):
    """Check whether the answer contains expected keywords."""

    normalized_answer = normalize_text(answer)

    matched_keywords = [
        keyword
        for keyword in expected_keywords
        if normalize_text(keyword) in normalized_answer
    ]

    missing_keywords = [
        keyword
        for keyword in expected_keywords
        if keyword not in matched_keywords
    ]

    score = (
        len(matched_keywords) / len(expected_keywords)
        if expected_keywords else 0.0
    )

    return score, matched_keywords, missing_keywords


# --------------------------------------------------
# Rate-limit detection and retry
# --------------------------------------------------

def ask_with_retry(graph, question, thread_id):
    """Call chatbot with retry protection for rate limits."""

    for attempt in range(1, MAX_RETRIES + 1):

        try:
            return ask(graph, question, thread_id)

        except Exception as error:

            error_message = str(error).lower()

            is_rate_limit = any(
                term in error_message
                for term in [
                    "rate limit",
                    "rate_limit",
                    "429",
                    "too many requests",
                    "tokens per minute",
                    "tpm",
                ]
            )

            if not is_rate_limit or attempt == MAX_RETRIES:
                raise

            wait_time = 20 * attempt

            print(
                f"Rate limit detected. "
                f"Retry {attempt}/{MAX_RETRIES} "
                f"after {wait_time} seconds..."
            )

            time.sleep(wait_time)

    raise RuntimeError("Maximum retries exceeded.")


# --------------------------------------------------
# Run evaluation
# --------------------------------------------------

def run_evaluation():
    """Run chatbot evaluation and log metrics to MLflow."""

    configure_mlflow()

    test_cases = load_test_cases()
    graph = build_graph()

    results = []
    passed = 0
    failed = 0

    model_name = os.getenv("MODEL", "unknown")

    with mlflow.start_run(run_name="rag-evaluation"):

        mlflow.log_params({
            "model": model_name,
            "embedding_model": "BAAI/bge-small-en-v1.5",
            "chunk_size": 800,
            "chunk_overlap": 100,
            "top_k": TOP_K,
            "total_test_cases": len(test_cases),
            "request_delay_seconds": REQUEST_DELAY_SECONDS,
            "max_retries": MAX_RETRIES,
        })

        for index, case in enumerate(test_cases, start=1):

            question = case["question"]
            expected_keywords = case["keywords"]

            print(
                f"\n[{index}/{len(test_cases)}] {question}"
            )

            start_time = time.perf_counter()

            try:
                response = ask_with_retry(
                    graph,
                    question,
                    str(uuid.uuid4()),
                )

                answer = response["answer"]
                route = response.get("route", "unknown")
                sources = response.get("sources", [])

                score, matched, missing = evaluate_answer(
                    answer,
                    expected_keywords,
                )

                route_correct = (
                    "expect_route" not in case
                    or route == case["expect_route"]
                )

                is_correct = score == 1.0 and route_correct

                status = "PASS" if is_correct else "FAIL"

                if is_correct:
                    passed += 1
                else:
                    failed += 1

                latency = time.perf_counter() - start_time

                result = {
                    "question": question,
                    "answer": answer,
                    "expected_keywords": expected_keywords,
                    "matched_keywords": matched,
                    "missing_keywords": missing,
                    "keyword_score": round(score, 3),
                    "route": route,
                    "route_correct": route_correct,
                    "sources_count": len(sources),
                    "latency_seconds": round(latency, 3),
                    "status": status,
                }

                results.append(result)

                print(f"Status: {status}")
                print(f"Route: {route}")
                print(f"Keyword score: {score:.2f}")
                print(f"Matched: {matched}")
                print(f"Missing: {missing}")
                print(f"Latency: {latency:.2f}s")

            except Exception as error:

                latency = time.perf_counter() - start_time
                failed += 1

                result = {
                    "question": question,
                    "expected_keywords": expected_keywords,
                    "status": "ERROR",
                    "error": str(error),
                    "latency_seconds": round(latency, 3),
                }

                results.append(result)

                print(
                    f"ERROR: {type(error).__name__}: {error}"
                )

            finally:
                if index < len(test_cases):
                    print(
                        f"Waiting {REQUEST_DELAY_SECONDS} seconds "
                        "before next question..."
                    )
                    time.sleep(REQUEST_DELAY_SECONDS)

        # ------------------------------------------
        # Aggregate evaluation metrics
        # ------------------------------------------

        accuracy = passed / len(test_cases)

        average_score = (
            mean([
                item.get("keyword_score", 0.0)
                for item in results
            ])
            if results else 0.0
        )

        valid_latencies = [
            item["latency_seconds"]
            for item in results
            if "latency_seconds" in item
        ]

        average_latency = (
            mean(valid_latencies)
            if valid_latencies else 0.0
        )

        mlflow.log_metrics({
            "accuracy": accuracy,
            "pass_count": passed,
            "fail_count": failed,
            "average_keyword_score": average_score,
            "average_latency_seconds": average_latency,
        })

        # ------------------------------------------
        # Save evaluation results
        # ------------------------------------------

        with RESULTS_FILE.open("w", encoding="utf-8") as file:
            json.dump(
                results,
                file,
                indent=4,
                ensure_ascii=False,
            )

        mlflow.log_artifact(str(RESULTS_FILE))

        # ------------------------------------------
        # Print summary
        # ------------------------------------------

        print("\n" + "=" * 50)
        print("EVALUATION SUMMARY")
        print("=" * 50)

        print(f"Total questions : {len(test_cases)}")
        print(f"Passed          : {passed}")
        print(f"Failed          : {failed}")
        print(f"Accuracy        : {accuracy:.2%}")
        print(f"Average score   : {average_score:.2%}")
        print(f"Average latency : {average_latency:.2f}s")
        print(f"Results saved   : {RESULTS_FILE}")

        print("=" * 50)


if __name__ == "__main__":
    run_evaluation()
