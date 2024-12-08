import subprocess
import os
import re
import json
import tempfile
from typing import List, Dict, Any, Optional

from MLAgentBench.LLM import complete_text, complete_text_fast, LOG_DIR

FEEDBACK_MODEL = "o1-mini"
FEEDBACK_MAX_TOKENS = 4000
MAX_ITERATIONS = 3
def call_llm(prompt):
    completion = complete_text(prompt, log_file=os.path.join(LOG_DIR, "env_log", "test_cases.txt"), model=FEEDBACK_MODEL, max_tokens_to_sample=FEEDBACK_MAX_TOKENS)
    return completion

def identify_method(proposal: str) -> str:
    """Extract the methodology from the proposal using LLM."""
    prompt = (
        "Extract only the methodology section from the following research proposal. "
        "Ignore abstract, motivation, impact, etc.\n\nMethod:\n" + proposal
    )
    return call_llm(prompt)


def identify_code(code: str) -> str:
    """Extract the core implementation from the provided code."""
    prompt = (
        "Extract the core implementation from the following code, ignoring class initialization, "
        "dataset/model/tokenizer loading, and other non-essential parts.\n\nCode:\n" + code
    )
    return call_llm(prompt)


def extract_python(response):
    match = re.search(r'```python(.*?)```', response, re.DOTALL)
    if match:
        code_content = match.group(1).strip()  # Extract Python code
        return code_content
    return None

def extract_json(response):
    match = re.search(r'```json(.*?)```', response, re.DOTALL)
    if match:
        code_content = match.group(1).strip()  # Extract Python code
        return code_content
    return None

def generate_test_cases(method: str, core_code: str) -> List[Dict[str, Any]]:
    """Generate test cases as a JSON array using LLM."""
    prompt = (
        "Given the propsoed method and its code implementation, generate a JSON array where each item "
        "contains the fields 'test_case' (a descriptive file name for the test case) and 'code' (the Python test code).\n\n"
        f"Method:\n{method}\n\nCode:\n```python\n{core_code}\n```\n\n\n"
        f"Each individual test case should be a self-contained file without any placeholders, including necessary package imports and class definitions. "
        "Each test function must adhere to pytest conventions and be properly formatted for execution. "
        "Write test functions that include assertions with error messages. Each error message should describe the condition being checked. Provide at least 10 test cases.\n"
    )
    for i in range(MAX_ITERATIONS):
        response = call_llm(prompt)
        try:
            test_cases = json.loads(extract_json(response))
            # Validate the format of test cases
            if all("test_case" in case and "code" in case for case in test_cases):
                return test_cases
        except Exception as e:
            continue
    return None


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
        result = subprocess.run(["pytest", file_path], capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired as e:
        return {"status": "ERROR", "message": f"Timeout!\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"} # RuntimeError
    is_assertion_error = "AssertionError" in result.stdout or "AssertionError" in result.stderr
    if result.returncode == 0:
        return {"status": "PASSED", "message": f"STDOUT:\n{result.stdout}"}
    elif is_assertion_error:
        return {"status": "FAILED", "message": f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"} # AssertionError
    else:
        return {"status": "ERROR", "message": f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"} # RuntimeError


def debug_test_case(test_case: Dict[str, Any], error_message: str, method: str, core_code: str) -> Optional[str]:
    """Attempt to debug a failing test case using LLM."""
    debug_result = None
    for i in range(MAX_ITERATIONS):
        prompt = (
            "The following test case failed with a runtime error. Modify the code to fix the issue and "
            "ensure it executes without errors. Do not fix assertion errors.\n\n"
            f"Test Case Code:\n```python\n{test_case['code']}\n```\n\nError Message:\n{error_message}\n\n"
            f"To provide some context, the test case is designed to verify the functionality of the following code:\n```python\n{core_code}\n```"
        )

        response = call_llm(prompt)
        try:
            # Extract and rewrite the test case code
            debugged_code = extract_python(response) # .split("```")[1]  # Extract code within triple backticks
            test_case["code"] = debugged_code
            debug_result = execute_test_case(test_case)
            if debug_result["status"] != "ERROR":
                return debug_result, test_case
            else:
                error_message = debug_result["message"]
            # return debug_result["output"] if debug_result["status"] == "PASSED" else debug_result["message"]
        except Exception as e:
            continue  # Could not extract valid code
    return debug_result, test_case


def process_test_cases(test_cases: List[Dict[str, Any]], core_code: str, method: str) -> List[Dict[str, Any]]:
    """Process test cases: run, debug, and collect results."""
    results = []

    for test_case in test_cases:
        result = execute_test_case(test_case)
        if result["status"] == "ERROR":  # Runtime error, attempt debugging
            debug_result, test_case = debug_test_case(test_case, result["message"], method, core_code)
            if debug_result:
                result = debug_result
        result["test_case"] = test_case
        results.append(result)

    return results


def summarize_results(results: List[Dict[str, Any]], core_code: str, method: str) -> None:
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
    if all_case_without_error == 0:
        pass_rate = None
    else:
        pass_rate = round(100 * passed_case / all_case_without_error, 2)
    if message:
        message = "===== Test Summary =====" + message
    pass_rate_stats = {
            "pass_rate" : pass_rate,
            "total_cases" : len(results), 
            "all_cases_without_error" : all_case_without_error,
            "passed_cases" : passed_case,
            }
    return {"test_case_pass_rate" : pass_rate_stats, "test_case_message" : message}


def test_cases_evaluation(full_method: str, full_code: str):
    """Main function to orchestrate the testing pipeline."""
    # method_code_pairs is deprecated
    method = identify_method(full_method)
    core_code = identify_code(full_code) 
    test_cases = generate_test_cases(method, core_code)
    if test_cases is None:
        return None
    results = process_test_cases(test_cases, core_code, method)
    test_cases_evaluation_result = summarize_results(results, core_code, method)
    return test_cases_evaluation_result
