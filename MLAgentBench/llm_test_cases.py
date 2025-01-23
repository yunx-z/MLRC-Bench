import subprocess
import os
import re
import json
import tempfile
from tqdm import tqdm
from typing import List, Dict, Any, Optional

from MLAgentBench.LLM import complete_text, complete_text_fast, LOG_DIR

FEEDBACK_MODEL = "o1-mini"
FEEDBACK_MAX_TOKENS = 4000
MAX_ITERATIONS = 1

os.makedirs(os.path.join(LOG_DIR, "env_log"), exist_ok=True)

def call_llm(prompt):
    for i in range(MAX_ITERATIONS):
        try:
            completion = complete_text(prompt, log_file=os.path.join(LOG_DIR, "env_log", "test_cases.txt"), model=FEEDBACK_MODEL, max_tokens_to_sample=FEEDBACK_MAX_TOKENS)
            return completion
        except Exception as e:
            continue
    return ""

def call_llm_fast(prompt):
    for i in range(MAX_ITERATIONS):
        try:
            completion = complete_text_fast(prompt, log_file=os.path.join(LOG_DIR, "env_log", "test_cases.txt"), max_tokens_to_sample=FEEDBACK_MAX_TOKENS)
            return completion
        except Exception as e:
            continue
    return ""


def identify_method(proposal: str) -> str:
    """Extract the methodology from the proposal using LLM."""
    prompt = (
        "Extract only the methodology section from the following research proposal. "
        "Ignore abstract, motivation, impact, etc.\n\nMethod:\n" + proposal
    )
    return call_llm_fast(prompt)


def identify_code(code: str) -> str:
    """Extract the core implementation from the provided code."""
    prompt = (
        "Extract the core implementation from the following code, ignoring class initialization, "
        "dataset/model/tokenizer loading, and other non-essential parts. Only include the code in response and nothing else.\n\n\n" + code
    )

    for i in range(MAX_ITERATIONS):
        res = extract_python(call_llm_fast(prompt))
        if res:
            return res
    return None


def extract_python(response):
    match = re.search(r'```python(.*?)```', response, re.DOTALL)
    if match:
        code_content = match.group(1).strip()  # Extract Python code
        return code_content
    return None

def extract_json(response):
    match = re.search(r'```json(.*?)```', response, re.DOTALL)
    if match:
        code_content = match.group(1).strip()
        return code_content
    return None


def generate_test_cases_iterative(method: str, core_code: str, iterations: int = 3, test_cases_per_iteration: int = 10) -> List[Dict[str, Any]]:
    """
    Iteratively generate test cases in multiple rounds. Each round conditions the LLM on
    previously generated test cases to produce new, more diverse ones.

    Parameters:
        method: The extracted methodology text.
        core_code: The extracted core code implementation.
        iterations: How many rounds of generation to perform.
        test_cases_per_iteration: How many test cases to generate per iteration.

    Returns:
        A list of test case dictionaries.
    """

    all_test_cases = []

    for i in tqdm(range(iterations), desc="Iteration"):
        # Include previously generated test cases in the prompt to ensure diversity
        if all_test_cases:
            # We'll show just the test case names and code from previous iterations
            previously_generated_str = json.dumps(all_test_cases, indent=2)
            prev_context = f"Previously generated test cases:\n```json\n{previously_generated_str}\n```\n"
        else:
            prev_context = "No previous test cases generated yet.\n"

        prompt = (
            f"Given the proposed method and its code implementation, generate a JSON array of {test_cases_per_iteration} new test cases. "
            "Avoid repeating previous test cases, ensuring new names and scenarios. "
            "Each item should contain the fields 'test_case' (a descriptive name without whitespace) and 'code' (the Python test code). "
            f"Method:\n{method}\n\nCode:\n```python\n{core_code}\n```\n\n"
            f"{prev_context}\n"
            "Each individual test case should be self-contained, with necessary imports and definitions included. Do not put any placeholders as the test cases will be directly executed without modifications by users. "
            "Use pytest conventions with assert statements that include error messages. "
            f"Output the test cases in strictly valid JSON format. No additional text outside of the JSON.\n"
        )

        response = call_llm(prompt)
        json_str = extract_json(response)
        if not json_str:
            # If we fail to extract JSON, continue to the next iteration
            continue

        try:
            test_cases = json.loads(json_str)
            # Validate the format of test cases
            if all("test_case" in case and "code" in case for case in test_cases):
                # Merge them with previously generated ones
                all_test_cases.extend(test_cases)
            else:
                # If format isn't correct, continue to next iteration
                continue
        except:
            # JSON parsing error, continue
            continue

    return all_test_cases if all_test_cases else None


def write_test_case_to_file(test_case: Dict[str, Any], output_dir: str = "test_cases/") -> str:
    """Write a test case to a separate file."""
    os.makedirs(output_dir, exist_ok=True)
    test_case_file = os.path.join(output_dir, f"{test_case['test_case']}.py")
    with open(test_case_file, "w") as file:
        file.write(test_case["code"])
    return test_case_file


def execute_test_case(test_case: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a test case using pytest and return the result."""
    file_path = write_test_case_to_file(test_case)
    try:
        result = subprocess.run(["pytest", file_path], capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired as e:
        return {"status": "ERROR", "message": f"Timeout!\nSTDOUT:\n{e.stdout}\n\nSTDERR:\n{e.stderr}"}
    is_assertion_error = "AssertionError" in result.stdout or "AssertionError" in result.stderr
    if result.returncode == 0:
        return {"status": "PASSED", "message": f"STDOUT:\n{result.stdout}"}
    elif is_assertion_error:
        return {"status": "FAILED", "message": f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"}
    else:
        return {"status": "ERROR", "message": f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"}


def debug_test_case(test_case: Dict[str, Any], error_message: str, method: str, core_code: str) -> Optional[str]:
    """Attempt to debug a failing test case using LLM."""
    debug_result = None
    for i in range(MAX_ITERATIONS):
        prompt = (
            "The following test case failed with a runtime error. Modify the code to fix the issue and "
            "ensure it executes without errors. Do not fix assertion errors.\n\n"
            f"Test Case Code:\n```python\n{test_case['code']}\n```\n\nError Message:\n{error_message}\n\n"
            f"Context code:\n```python\n{core_code}\n```"
        )

        response = call_llm(prompt)
        debugged_code = extract_python(response)
        if debugged_code:
            test_case["code"] = debugged_code
            debug_result = execute_test_case(test_case)
            if debug_result["status"] != "ERROR":
                return debug_result, test_case
            else:
                error_message = debug_result["message"]
    return debug_result, test_case


def process_test_cases(test_cases: List[Dict[str, Any]], core_code: str, method: str) -> List[Dict[str, Any]]:
    """Process test cases: run, debug, and collect results."""
    results = []

    for test_case in tqdm(test_cases, desc='Execution'):
        result = execute_test_case(test_case)
        if result["status"] == "ERROR":  # Runtime error, attempt debugging
            debug_result, test_case = debug_test_case(test_case, result["message"], method, core_code)
            if debug_result:
                result = debug_result
        result["test_case"] = test_case
        results.append(result)

    return results


def summarize_results(results: List[Dict[str, Any]], core_code: str, method: str) -> Dict[str, Any]:
    """Summarize and display test results."""
    all_case_without_error, passed_case = 0, 0
    message = ""
    for result in results:
        if result["status"] == "FAILED":
            message += f"\nTest Case:\n```python\n{result['test_case']['code']}\n```\n\n"
            message += f"Status: {result['status']}\n"
            message += f"Message:\n{result['message']}\n"
        if result["status"] != "ERROR":
            all_case_without_error += 1
        if result["status"] == "PASSED":
            passed_case += 1
    pass_rate = None if all_case_without_error == 0 else round(100 * passed_case / all_case_without_error, 2)

    if message:
        message = "===== Test Summary =====" + message
    pass_rate_stats = {
        "pass_rate": pass_rate,
        "total_cases": len(results),
        "all_cases_without_error": all_case_without_error,
        "passed_cases": passed_case,
    }
    return {"test_case_pass_rate": pass_rate_stats, "test_case_message": message}


def test_cases_evaluation(full_method: str, full_code: str, num_iterations: int = 5, test_cases_per_iteration: int = 10):
    """Main function to orchestrate the testing pipeline with iterative generation."""
    method = identify_method(full_method)
    core_code = identify_code(full_code)
    if core_code is None:
        return None

    # Use the iterative generation approach
    test_cases = generate_test_cases_iterative(method, core_code, iterations=num_iterations, test_cases_per_iteration=test_cases_per_iteration)
    if test_cases is None:
        return None

    results = process_test_cases(test_cases, core_code, method)
    test_cases_evaluation_result = summarize_results(results, core_code, method)
    return test_cases_evaluation_result

