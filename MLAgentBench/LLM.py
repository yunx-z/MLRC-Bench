""" This file contains the code for calling all LLM APIs. """

import os
import time
import re
import json
from .schema import TooLongPromptError, LLMError


# https://openai.com/api/pricing/ as of 01/10/2025
# https://aws.amazon.com/bedrock/pricing/ as of 01/20/2025
# https://ai.google.dev/pricing#1_5pro as of 01/20/2025
MODEL2PRICE = {
        "gpt-4o" : {
            "input" : 2.5 / 1e6,
            "output" : 10 / 1e6,
            },
        "gpt-4o-mini" : {
            "input" : 0.15 / 1e6,
            "output" : 0.6 / 1e6,
            },
        "o1-mini" : {
            "input" : 3 / 1e6,
            "output" : 12 / 1e6,
            },
        "o1-preview" : {
            "input" : 15 / 1e6,
            "output" : 60 / 1e6,
            },
        "o1" : {
            "input" : 15 / 1e6,
            "output" : 60 / 1e6,
            },
        "claude-3-5-sonnet-v2" : {
            "input" : 0.003 / 1000,
            "output" : 0.015 / 1000,
            },
        "claude-3-5-haiku" : {
            "input" : 0.0008 / 1000,
            "output" : 0.004 / 1000,
            },
        "claude-3-opus" : {
            "input" : 0.015 / 1000,
            "output" : 0.075 / 1000,
            },
        "gemini-exp-1206" : {
            "input" : 0,
            "output" : 0,
            },
        "gemini-2.0-flash-thinking-exp-0121" : {
            "input" : 0,
            "output" : 0,
            },
        "gemini-2.0-flash-exp" : {
            "input" : 0,
            "output" : 0,
            },
        "gemini-1.5-pro-002" : {
            "input" : 1.25 / 1e6,
            "output" : 5 / 1e6,
            },
        "gemini-1.5-flash-002" : {
            "input" : 0.075 / 1e6,
            "output" : 0.3 / 1e6,
            },
        "llama3-1-405b-instruct" : {
            "input" : 0.0024 / 1000,
            "output" : 0.0024 / 1000,
            },
        "llama3-3-70b-instruct" : {
            "input" : 0.00072 / 1000,
            "output" : 0.00072 / 1000,
            },
        "DeepSeek-R1" : {
            "input" : 0,
            "output" : 0,
            },
        }


try:
    from helm.common.authentication import Authentication
    from helm.common.request import Request, RequestResult
    from helm.proxy.accounts import Account
    from helm.proxy.services.remote_service import RemoteService
    # setup CRFM API
    auth = Authentication(api_key=open("crfm_api_key.txt").read().strip())
    service = RemoteService("https://crfm-models.stanford.edu")
    account: Account = service.get_account(auth)
except Exception as e:
    pass
    # print(e)
    # print("Could not load CRFM API key crfm_api_key.txt.")

try:   
    import anthropic
    import boto3
    from botocore.exceptions import ClientError
    from botocore.config import Config

    my_config = Config(read_timeout=1000)

    bedrock_client = boto3.client(service_name='bedrock-runtime', region_name='us-west-2', aws_access_key_id=os.environ["AWS_ACCESS_KEY"], aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"], config=my_config)
    BEDROCK_MODEL_IDS = {
        "claude-3-5-sonnet-v2" : "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        "claude-3-5-haiku" : "us.anthropic.claude-3-5-haiku-20241022-v1:0",
        "claude-3-opus" : "us.anthropic.claude-3-opus-20240229-v1:0",
        "llama3-3-70b-instruct" : "us.meta.llama3-3-70b-instruct-v1:0",
        "llama3-1-405b-instruct" : "meta.llama3-1-405b-instruct-v1:0",
        }
except Exception as e:
    pass
    # print(e)
    # print("Could not load anthropic API key claude_api_key.txt.")

try:
    import openai
    # setup OpenAI API key
    openai_api_key = os.getenv('MY_OPENAI_API_KEY')
    openai_api_base = os.getenv('MY_AZURE_OPENAI_ENDPOINT')
    openai_client = openai.AzureOpenAI(
            azure_endpoint=openai_api_base,
            api_key=openai_api_key,
            api_version="2024-12-01-preview",
            )
except Exception as e:
    pass

try:
    from azure.ai.inference import ChatCompletionsClient
    from azure.core.credentials import AzureKeyCredential
    from azure.ai.inference.models import SystemMessage, UserMessage

    azure_ai_client = ChatCompletionsClient(
	endpoint=os.environ["AZUREAI_ENDPOINT_URL"],
	credential=AzureKeyCredential(os.environ["AZUREAI_ENDPOINT_KEY"]),
        api_version="2024-05-01-preview",
    )
except Exception as e:
    pass

try:
    import vertexai
    from vertexai.preview.generative_models import GenerativeModel, Part
    from google.cloud.aiplatform_v1beta1.types import SafetySetting, HarmCategory
    vertexai.init(project=os.environ["GCP_PROJECT_ID"], location="us-central1")
except Exception as e:
    pass

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers import StoppingCriteria, StoppingCriteriaList
    import torch

    loaded_hf_models = {}

    class StopAtSpecificTokenCriteria(StoppingCriteria):
        def __init__(self, stop_sequence):
            super().__init__()
            self.stop_sequence = stop_sequence

        def __call__(self, input_ids, scores, **kwargs):
            # Create a tensor from the stop_sequence
            stop_sequence_tensor = torch.tensor(self.stop_sequence, device=input_ids.device, dtype=input_ids.dtype)

            # Check if the current sequence ends with the stop_sequence
            current_sequence = input_ids[:, -len(self.stop_sequence) :]
            return bool(torch.all(current_sequence == stop_sequence_tensor).item())
except Exception as e:
    pass

def log_to_file(log_file, prompt, completion, model, max_tokens_to_sample, num_prompt_tokens, num_sample_tokens, thought=None):
    """ Log the prompt and completion to a file."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, "a") as f:
        f.write("\n===================prompt=====================\n")
        f.write(f"{prompt}")
        if thought:
            f.write(f"\n==================={model} thought ({max_tokens_to_sample})=====================\n")
            f.write(f"{thought}")
        f.write(f"\n==================={model} response ({max_tokens_to_sample})=====================\n")
        f.write(completion)
        f.write("\n===================tokens=====================\n")
        f.write(f"Number of prompt tokens: {num_prompt_tokens}\n")
        f.write(f"Number of sampled tokens: {num_sample_tokens}\n")
        f.write("\n\n")

    LOG_DIR = os.getenv("LOG_DIR", "logs/")
    cost_file = os.path.join(LOG_DIR, "env_log/", "api_cost.json")
    content = dict()
    if os.path.exists(cost_file):
        with open(cost_file, 'r') as reader:
            content = json.load(reader)

    curr_cost = num_prompt_tokens * MODEL2PRICE[model]["input"] + num_sample_tokens * MODEL2PRICE[model]["output"] 
    with open(cost_file, 'w') as writer:
        updated_content = {
                "total_cost" : content.get("total_cost", 0) + curr_cost,
                "total_num_prompt_tokens" : content.get("total_num_prompt_tokens", 0) + num_prompt_tokens,
                "total_num_sample_tokens" : content.get("total_num_sample_tokens", 0) + num_sample_tokens,
                }
        json.dump(updated_content, writer, indent=2)


def complete_text_hf(prompt, stop_sequences=[], model="huggingface/codellama/CodeLlama-7b-hf", max_tokens_to_sample = 4000, temperature=0.5, log_file=None, **kwargs):
    model = model.split("/", 1)[1]
    if model in loaded_hf_models:
        hf_model, tokenizer = loaded_hf_models[model]
    else:
        hf_model = AutoModelForCausalLM.from_pretrained(model).to("cuda:9")
        tokenizer = AutoTokenizer.from_pretrained(model)
        loaded_hf_models[model] = (hf_model, tokenizer)
        
    encoded_input = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to("cuda:9")
    stop_sequence_ids = tokenizer(stop_sequences, return_token_type_ids=False, add_special_tokens=False)
    stopping_criteria = StoppingCriteriaList()
    for stop_sequence_input_ids in stop_sequence_ids.input_ids:
        stopping_criteria.append(StopAtSpecificTokenCriteria(stop_sequence=stop_sequence_input_ids))

    output = hf_model.generate(
        **encoded_input,
        temperature=temperature,
        max_new_tokens=max_tokens_to_sample,
        do_sample=True,
        return_dict_in_generate=True,
        output_scores=True,
        stopping_criteria = stopping_criteria,
        **kwargs,
    )
    sequences = output.sequences
    sequences = [sequence[len(encoded_input.input_ids[0]) :] for sequence in sequences]
    all_decoded_text = tokenizer.batch_decode(sequences)
    completion = all_decoded_text[0]
    completion = re.sub(r'\*\*(.*?)\*\*', r'\1', completion)
    if log_file is not None:
        log_to_file(log_file, prompt, completion, model, max_tokens_to_sample)
    return completion


def complete_text_gemini(prompt, stop_sequences=[], model="gemini-pro", max_tokens_to_sample = 8000, temperature=0.5, log_file=None, **kwargs):
    """ Call the gemini API to complete a prompt."""
    # Load the model
    gemini_model = GenerativeModel(model)
    # Query the model
    parameters = {
            "temperature": temperature,
            "max_output_tokens": max_tokens_to_sample,
            "stop_sequences": stop_sequences,
            **kwargs
        }
    safety_settings = {
            harm_category: SafetySetting.HarmBlockThreshold(SafetySetting.HarmBlockThreshold.BLOCK_NONE)
            for harm_category in iter(HarmCategory)
        }
    response = gemini_model.generate_content( [prompt], generation_config=parameters, safety_settings=safety_settings)
    if "thinking" in model:
        print(response)
        thought = response.candidates[0].content.parts[0].text
        completion = response.candidates[0].content.parts[1].text
    else:
        thought = None
        completion = response.text
    completion = re.sub(r'\*\*(.*?)\*\*', r'\1', completion)
    num_prompt_tokens = response.usage_metadata.prompt_token_count
    num_sample_tokens = response.usage_metadata.candidates_token_count
    if log_file is not None:
        log_to_file(log_file, prompt, completion, model, max_tokens_to_sample, num_prompt_tokens, num_sample_tokens, thought=thought)
    return completion

def complete_text_claude(prompt, stop_sequences=None, model="claude-v1", max_tokens_to_sample = 8000, temperature=0.5, log_file=None, messages=None, **kwargs):
    """ Call the Claude API to complete a prompt."""
    if stop_sequences is None:
        stop_sequences = [anthropic.HUMAN_PROMPT]



    model_id = BEDROCK_MODEL_IDS[model]
    native_request = {
	"anthropic_version": "bedrock-2023-05-31",
	"max_tokens": max_tokens_to_sample,
	"temperature": temperature,
        "stop_sequences": stop_sequences,
	"messages": [
	    {
		"role": "user",
		"content": [{"type": "text", "text": prompt}],
	    }
	],
    }

    # Convert the native request to JSON.
    request = json.dumps(native_request)

    # Invoke the model with the request.
    response = bedrock_client.invoke_model(modelId=model_id, body=request)

    # Decode the response body.
    model_response = json.loads(response["body"].read())
    # model_response {'id': 'msg_bdrk_01HmMRoLhxeydUYyyrsSCn5R', 'type': 'message', 'role': 'assistant', 'model': 'claude-3-5-haiku-20241022', 'content': [{'type': 'text', 'text': 'Hi there! How are you doing today? Is there anything I can help you with?'}], 'stop_reason': 'end_turn', 'stop_sequence': None, 'usage': {'input_tokens': 9, 'output_tokens': 23}}

    # Extract and print the response text.
    completion = model_response["content"][0]["text"]
    completion = re.sub(r'\*\*(.*?)\*\*', r'\1', completion)
    num_prompt_tokens = model_response["usage"]["input_tokens"]
    num_sample_tokens = model_response["usage"]["output_tokens"]
    
    if log_file is not None:
        log_to_file(log_file, prompt, completion, model, max_tokens_to_sample, num_prompt_tokens, num_sample_tokens)
    return completion

def complete_text_llama(prompt, stop_sequences=None, model="llama3-3-70b-instruct", max_tokens_to_sample = 4000, temperature=0.5, log_file=None, messages=None, **kwargs):
    model_id = BEDROCK_MODEL_IDS[model]
    # Embed the prompt in Llama 3's instruction format.
    formatted_prompt = f"""
    <|begin_of_text|><|start_header_id|>user<|end_header_id|>
    {prompt}
    <|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """
    native_request = {
	"prompt": formatted_prompt,
	"max_gen_len": max_tokens_to_sample,
	"temperature": temperature,
    }

    # Convert the native request to JSON.
    request = json.dumps(native_request)

    # Invoke the model with the request.
    response = bedrock_client.invoke_model(modelId=model_id, body=request)

    # Decode the response body.
    model_response = json.loads(response["body"].read())
    # model_response {'id': 'msg_bdrk_01HmMRoLhxeydUYyyrsSCn5R', 'type': 'message', 'role': 'assistant', 'model': 'claude-3-5-haiku-20241022', 'content': [{'type': 'text', 'text': 'Hi there! How are you doing today? Is there anything I can help you with?'}], 'stop_reason': 'end_turn', 'stop_sequence': None, 'usage': {'input_tokens': 9, 'output_tokens': 23}}

    # Extract and print the response text.
    completion = model_response["generation"]
    completion = re.sub(r'\*\*(.*?)\*\*', r'\1', completion)
    # Since bedrock's llama-series model does not support stop_sequences, we need to truncate observation by ourselves
    completion = re.sub(r"^Observation:.*", "", completion, flags=re.DOTALL | re.MULTILINE)

    num_prompt_tokens = model_response["prompt_token_count"]
    num_sample_tokens = model_response["generation_token_count"]
    
    if log_file is not None:
        log_to_file(log_file, prompt, completion, model, max_tokens_to_sample, num_prompt_tokens, num_sample_tokens)
    return completion



def get_embedding_crfm(text, model="openai/gpt-4-0314"):
    request = Request(model="openai/text-embedding-ada-002", prompt=text, embedding=True)
    request_result: RequestResult = service.make_request(auth, request)
    return request_result.embedding 
    
def complete_text_crfm(prompt="", stop_sequences = [], model="openai/gpt-4-0314",  max_tokens_to_sample=4000, temperature = 0.5, log_file=None, messages = None, **kwargs):
    
    random = log_file
    if messages:
        request = Request(
                prompt=prompt, 
                messages=messages,
                model=model, 
                stop_sequences=stop_sequences,
                temperature = temperature,
                max_tokens = max_tokens_to_sample,
                random = random
            )
    else:
        # print("model", model)
        # print("max_tokens", max_tokens_to_sample)
        request = Request(
                # model_deployment=model,
                prompt=prompt, 
                model=model, 
                stop_sequences=stop_sequences,
                temperature = temperature,
                max_tokens = max_tokens_to_sample,
                random = random
        )
    
    try:      
        request_result: RequestResult = service.make_request(auth, request)
    except Exception as e:
        # probably too long prompt
        print(e)
        raise TooLongPromptError()
    
    if request_result.success == False:
        print(request.error)
        raise LLMError(request.error)
    completion = request_result.completions[0].text
    completion = re.sub(r'\*\*(.*?)\*\*', r'\1', completion)
    if log_file is not None:
        log_to_file(log_file, prompt if not messages else str(messages), completion, model, max_tokens_to_sample)
    return completion


def complete_text_openai(prompt, stop_sequences=[], model="gpt-4o-mini", max_tokens_to_sample=4000, temperature=0.5, log_file=None, **kwargs):
    """ Call the OpenAI API to complete a prompt."""
    if "o1" in model.lower():
        raw_request = {
              "model": model,
              "temperature": 1,
              "max_completion_tokens": 64000 if model.lower() == "o1-mini" else 32000,
              **kwargs
        }
    else:
        raw_request = {
              "model": model,
              "temperature": temperature,
              "max_tokens": max_tokens_to_sample,
              "stop": stop_sequences or None,  # API doesn't like empty list
              **kwargs
        }

    messages = [{"role": "user", "content": prompt}]
    response = openai_client.chat.completions.create(**{"messages": messages,**raw_request})
    completion = response.choices[0].message.content
    usage = response.usage

    
    completion = re.sub(r'\*\*(.*?)\*\*', r'\1', completion)

    # Since o1-series model does not support stop_sequences, we need to truncate observation by ourselves
    if "o1" in model.lower():
        completion = re.sub(r"^Observation:.*", "", completion, flags=re.DOTALL | re.MULTILINE)

    if log_file is not None:
        log_to_file(log_file, prompt, completion, model, max_tokens_to_sample, num_prompt_tokens=usage.prompt_tokens, num_sample_tokens=usage.completion_tokens)
    return completion

def separate_thought_completion(text):
    match = re.search(r'<think>(.*?)</think>(.*)', text, re.DOTALL)
    if match:
        thought = match.group(1).strip()
        completion = match.group(2).strip()
        return thought, completion
    return None, None

def complete_text_deepseek(prompt, stop_sequences=[], model="DeepSeek-R1", max_tokens_to_sample=4000, temperature=0.5, log_file=None, **kwargs):
    """ Call the OpenAI API to complete a prompt."""
    raw_request = {
          "temperature": temperature,
          "max_tokens": max_tokens_to_sample,
          "stop": stop_sequences or None,  # API doesn't like empty list
          **kwargs
    }

    response = azure_ai_client.complete(
        model=model,
	messages=[
	    UserMessage(content=prompt),
	],
	model_extras=raw_request,
    )
    thought_and_completion = response.choices[0].message.content
    usage = response.usage
    thought, completion = separate_thought_completion(thought_and_completion)

    completion = re.sub(r'\*\*(.*?)\*\*', r'\1', completion)

    if log_file is not None:
        log_to_file(log_file, prompt, completion, model, max_tokens_to_sample, num_prompt_tokens=usage.prompt_tokens, num_sample_tokens=usage.completion_tokens, thought=thought)
    return completion


MAX_RETRIES=10
WAIT_TIME=60
def complete_text(prompt, log_file, model, **kwargs):
    """ Complete text using the specified model with appropriate API. """
    assert log_file is not None, "log_file is None"

    retry = 0
    error_msg = None
    while retry < MAX_RETRIES:
        retry += 1
        try:
            if model.startswith("claude"):
                # use anthropic API
                completion = complete_text_claude(prompt, stop_sequences=[anthropic.HUMAN_PROMPT, "Observation:"], log_file=log_file, model=model, **kwargs)
            elif model.startswith("llama"):
                completion = complete_text_llama(prompt, stop_sequences=["Observation:"], log_file=log_file, model=model, **kwargs)
            elif model.startswith("gemini"):
                completion = complete_text_gemini(prompt, stop_sequences=["Observation:"], log_file=log_file, model=model, **kwargs)
            elif model.startswith("huggingface"):
                completion = complete_text_hf(prompt, stop_sequences=["Observation:"], log_file=log_file, model=model, **kwargs)
            elif "/" in model:
                # use CRFM API since this specifies organization like "openai/..."
                completion = complete_text_crfm(prompt, stop_sequences=["Observation:"], log_file=log_file, model=model, **kwargs)
            elif model.startswith("DeepSeek"):
                completion = complete_text_deepseek(prompt, stop_sequences=["Observation:"], log_file=log_file, model=model, **kwargs)
            else:
                # use OpenAI API
                completion = complete_text_openai(prompt, stop_sequences=["Observation:"], log_file=log_file, model=model, **kwargs)
            return completion
        except Exception as e:
            print(f"{model} attempt {retry} failed!")
            error_msg = e
            time.sleep(WAIT_TIME)

    raise LLMError(str(error_msg))

# specify fast models for summarization etc
def complete_text_fast(prompt, **kwargs):
    FAST_MODEL = os.getenv("FAST_MODEL", "gpt-4o-mini")
    return complete_text(prompt = prompt, model = FAST_MODEL, temperature =0.01, **kwargs)

if __name__ == "__main__":
    os.makedirs("logs/env_log", exist_ok=True)
    # for model in ["o1", "o1-mini", "gpt-4o", "gpt-4o-mini", "claude-3-5-sonnet-v2", "gemini-exp-1206", "llama3-1-405b-instruct"]:
    for model in ["DeepSeek-R1"]:
        completion = complete_text("12+32=?", "logs/tmp.log", model)
        print(model)
        print(completion)
