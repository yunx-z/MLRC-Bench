import json
import os
import time
from typing import Dict

from MLAgentBench.LLM import complete_text
from MLAgentBench.constants import MLR_BENCH_DIR

# adapted from https://arxiv.org/pdf/2404.07738
rubric = {
        "Clarity" : """
        It assesses whether the method is described in a clear, precise, and understandable manner that allows for
        replication and comprehension of the approach.
        1. The method is explained in an extremely vague or ambiguous manner, making it impossible to understand or
        replicate the approach without additional information or clarification.
        2. The method is described with some detail, but significant gaps in explanation or logic leave the reader with
        considerable confusion and uncertainty about how to apply or replicate the approach.
        3. The method is described with sufficient detail to understand the basic approach, but lacks the precision or
        specificity needed to fully replicate or grasp the nuances of the methodology without further guidance.
        4. The method is clearly and precisely described, with most details provided to allow for replication and
        comprehension, though minor areas may benefit from further clarification or elaboration.
        5. The method is articulated in an exceptionally clear, precise, and detailed manner, enabling straightforward
        replication and thorough understanding of the approach with no ambiguities.
        """,
        "Validity" : """
        It measures the accuracy, relevance, and soundness of the method in addressing the research problem, ensuring
        that it is appropriate and directly relevant to the objectives of the study.
        1. The method shows a fundamental misunderstanding of the research problem and lacks any credible alignment
        with established scientific principles or relevant studies.
        2. The method partially addresses the research problem but exhibits significant flaws in its scientific underpin-
        ning, making its validity questionable despite some alignment with existing literature.
        3. The method adequately addresses the research problem but with some limitations in its scientific validity,
        showing a mix of strengths and weaknesses in its alignment with related studies.
        4. The method effectively addresses the research problem, demonstrating a strong scientific basis and sound
        alignment with existing literature, albeit with minor areas for improvement.
        5. The method exemplifies an exceptional understanding of the research problem, grounded in a robust scientific
        foundation, and shows exemplary integration and advancement of existing studies’ findings.
        """,
        "Rigorousness" : """
        It examines the thoroughness, precision, and consistency of the method, ensuring that the approach is systematic,
        well-structured, and adheres to high standards of research quality.
        1. The method demonstrates a fundamental lack of systematic approach, with significant inconsistencies and
        inaccuracies in addressing the research problem, showing a disregard for established research standards.
        2. The method shows a minimal level of systematic effort but is marred by notable inaccuracies, lack of
        precision, and inconsistencies that undermine the rigorousness of the method in tackling the research problem.
        3. The method exhibits an average level of systematic structure and adherence to research standards but lacks
        the thoroughness, precision, and consistency required for a rigorous scientific inquiry.
        4. The method is well-structured and systematic, with a good level of precision and consistency, indicating a
        strong adherence to research standards, though it falls short of exemplifying the highest level of rigorousness.
        5. The method exemplifies exceptional rigorousness, with outstanding thoroughness, precision, and consistency
        in its systematic approach, setting a benchmark for high standards in scientific research quality.
        """,
        "Innovativeness" : """
        It evaluates whether the method introduces new techniques, approaches, or perspectives to the research field
        that differ from standard research practices and advance them in the field.
        1. The method introduces no novel elements, fully relying on existing techniques without any attempt to modify
        or adapt them for the specific research problem, showing a lack of innovativeness.
        2. The method shows minimal innovation, with only slight modifications to existing techniques that do not
        substantially change or improve the approach to the research problem.
        3. The method demonstrates moderate innovativeness, incorporating known techniques with some new elements
        or combinations that offer a somewhat fresh approach to the research problem but fall short of a significant
        breakthrough.
        4. The method is highly innovative, introducing new techniques or novel combinations of existing methods that
        significantly differ from standard practices, offering a new perspective or solution to the research problem.
        5. The method represents a groundbreaking innovation, fundamentally transforming the approach to the
        research problem with novel techniques or methodologies that redefine the field’s standard practices.
        """,
        "Generalizability" : """
        It assesses the extent to which the method can be applied to or is relevant for other contexts, populations, or
        settings beyond the scope of the study.
        1. The method shows no adaptability, failing to extend its applicability beyond its original context or dataset,
        showing a complete lack of generalizability.
        2. The method demonstrates minimal adaptability, with limited evidence of potential applicability to contexts
        slightly different from the original.
        3. The method exhibits some level of adaptability, suggesting it could be applicable to related contexts or
        datasets with modifications.
        4. The method is adaptable and shows evidence of applicability to a variety of contexts or datasets beyond the
        original.
        5. The method is highly adaptable, demonstrating clear evidence of broad applicability across diverse contexts,
        populations, and settings.
        """,
        }

def validate_json_response(response: str) -> Dict[str, str]:
    """
    Validates and parses a JSON response to ensure it contains the required fields and correct data types.

    Args:
        response (str): The JSON response string from the LLM.

    Returns:
        Dict[str, str]: The parsed and validated JSON object.

    Raises:
        ValueError: If the JSON is invalid or missing required fields.
    """
    parsed_response = json.loads(response)

    if not isinstance(parsed_response, dict):
        raise ValueError("Response is not a dictionary.")

    if not all(key in parsed_response for key in ["Review", "Feedback", "Rating"]):
        raise ValueError("Response is missing required fields.")

    if not isinstance(parsed_response["Review"], str):
        raise ValueError("Review field is not a string.")

    if not isinstance(parsed_response["Feedback"], str):
        raise ValueError("Feedback field is not a string.")

    if not (isinstance(parsed_response["Rating"], int) and 1 <= parsed_response["Rating"] <= 5):
        raise ValueError("Rating field is not an integer between 1 and 5.")

    return parsed_response

def llm_evaluate_method_on_metric(scientificMethod: str, code: str, researchProblem: str, metric: str, criteria: str, add_code=False, max_retries: int = 3, delay: int = 2) -> Dict[str, str]:
    """
    Evaluates a scientific method using an LLM, with retries in case of failure.

    Args:
        scientificMethod (str): The scientific method description.
        researchProblem (str): The research problem context.
        metric (str): The evaluation metric.
        criteria (str): Specific criteria for evaluation.
        max_retries (int): Maximum number of retry attempts for LLM call.
        delay (int): Delay (in seconds) between retries.

    Returns:
        Dict[str, str]: A JSON dictionary containing Review, Feedback, and Rating.
    """
    # Define the prompt for the LLM
    prompt = f"""
    You are an AI assistant whose primary goal is to assess the quality and soundness of scientific
    methods across diverse dimensions, in order to aid researchers in refining their methods based on
    your evaluations and feedback, thereby enhancing the impact and reach of their work.

    You are going to evaluate a scientific method for its {metric} in addressing a research problem,
    focusing on how well it is described in a clear, precise, and understandable manner that allows for
    replication and comprehension of the approach.

    As part of your evaluation, you can refer to the research problem, which will
    help in understanding the context of the proposed method for a more comprehensive assessment.

    Research problem: {researchProblem}

    Now, proceed with your {metric} evaluation approach that should be systematic:
    - Start by thoroughly reading the proposed method and its rationale, keeping in mind the context
      provided by the research problem, and existing studies mentioned above.
    - Next, generate a review and feedback that should be constructive, helpful, and concise, focusing
      on the {metric} of the method.
    - Finally, provide a score on a 5-point Likert scale, with 1 being the lowest, please ensuring a
      discerning and critical evaluation to avoid a tendency towards uniformly high ratings (4-5) unless
      fully justified:

    The criteria for {metric} evaluation: {criteria}
    
    """

    if add_code:
        prompt += f"""
    I am going to provide the proposed method with its code implementation, as follows:
    Proposed method: {scientificMethod}

    Code implementation:
    ```python
    {code}
    ```
    """
    else:
        prompt += f"""
    I am going to provide the proposed method as follows:
    Proposed method: {scientificMethod}
    ```
    """

    prompt += f"""
    After your evaluation of the above content, please respond **only** with a valid JSON object in the following format:
    {{
        "Review": "<Your review here>",
        "Feedback": "<Your feedback here>",
        "Rating": <Your rating here>
    }}
    """

    retries = 0
    log_file = os.path.join(os.getenv("LOG_DIR", "logs/"), "env_log", "llm_as_a_judge.txt")
    FEEDBACK_MODEL = os.getenv("FEEDBACK_MODEL", "o1")
    FEEDBACK_MAX_TOKENS = int(os.getenv("FEEDBACK_MAX_TOKENS", "4000"))
    while retries < max_retries:
        try:
            # Call the LLM
            response = complete_text(prompt, log_file=log_file, model=FEEDBACK_MODEL, max_tokens_to_sample=FEEDBACK_MAX_TOKENS)
            response = response.replace("```json", "").replace("```", "").strip()
            # Parse the response into the desired JSON format
            return validate_json_response(response)
        except Exception as e:
            retries += 1
            time.sleep(delay)
    return None

def llm_evaluate_method(scientificMethod: str, code: str, task_name: str, max_retries: int = 3, delay: int = 2) -> Dict[str, str]:
    research_problem_file = os.path.join(MLR_BENCH_DIR, f"MLAgentBench/benchmarks_base/{task_name}/scripts/research_problem.txt")
    with open(research_problem_file, 'r') as reader:
        researchProblem = reader.read()
    eval_result = {"with_code" : {}, "without_code" : {}}
    for metric in rubric:
        json_result = llm_evaluate_method_on_metric(scientificMethod, code, researchProblem, metric, rubric[metric], add_code=False)
        eval_result["without_code"][metric] = json_result
        json_result = llm_evaluate_method_on_metric(scientificMethod, code, researchProblem, metric, rubric[metric], add_code=True)
        eval_result["with_code"][metric] = json_result
    return eval_result

