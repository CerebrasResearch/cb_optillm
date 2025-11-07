from typing import Literal, Any, Optional
from optillm.cepo.cepo import cepo, CepoConfig


def cepo_tool_v3(messages: list, client: Any, model: str, request_config: dict = None) -> tuple[str, int]:
    cb_log = {}
    cb_log["cepo_version"] = 3
    cb_log["cepo_version_description"] = "Requesting the model to evaluate its confidence of the next step"
    completion_tokens = 0
    tools = request_config["tools"]

    step1_prompt = "\n\nLet's have an internal monologue before you decide on the next step interaction with the environment. To that end, can you state in natural language the following information (follow the format below):\ni) What do you want to do next and why,\nii) what is your confidence about the correctness of the next step in form of [[#]] where # is number from 0 to 10, 0 meaning no confidence at all, and 10 meaning maximum confidence,\niii) which tool would be a good choice to execute this next step?"

    messages[-1]["content"][0]["text"] = f"{messages[-1]["content"][0]["text"]}{step1_prompt}"

    step1_response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        #tool_choice=None,
        #max_tokens=6666
    )
    completion_tokens += step1_response.usage.completion_tokens
    print("--- step1 respose ---")
    print(step1_response.choices[0].finish_reason)
    print(step1_response.choices[0].message.content)
    cb_log["step1_prompt"] = step1_prompt
    cb_log["step1_response"] = step1_response.choices[0].message.content
    cb_log["step1_tool_calls"] = [t.model_dump() for t in step1_response.choices[0].message.tool_calls]
    cb_log["step1_finish_reason"] = step1_response.choices[0].finish_reason

    step2_prompt = "Can you execute the above step to generate the next interaction command for the environment"

    messages.append({"role": "assistant", "content": step1_response.choices[0].message.content})
    messages.append({"role": "user", "content": step2_prompt})

    step2_response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        #tool_choice="required",
        #max_tokens=6666
    )
    if step2_response.choices[0].message.content:
        step2_response.choices[0].message.content = f"I will take the following approach:\n{step1_response.choices[0].message.content}\n\nMy next step:\n{step2_response.choices[0].message.content}"
    else:
        step2_response.choices[0].message.content = step1_response.choices[0].message.content
    completion_tokens += step2_response.usage.completion_tokens
    print("--- step2 respose ---")
    print(step2_response.choices[0].finish_reason)
    print(step2_response.choices[0].message.content)
    cb_log["step2_prompt"] = step2_prompt
    cb_log["step2_finish_reason"] = step2_response.choices[0].finish_reason
    
    step2_response.choices[0].cb_log = cb_log
    return step2_response, completion_tokens


def cepo_tool_v4(messages: list, client: Any, model: str, request_config: dict = None) -> tuple[str, int]:
    cb_log = {}
    cb_log["cepo_version"] = 4
    cb_log["cepo_version_description"] = "Add tool calls to the assistant's text response in step 1"
    completion_tokens = 0
    tools = request_config["tools"]

    step1_prompt = "\n\nLet's have an internal monologue before you decide on the next step interaction with the environment. To that end, can you state in natural language the following information (follow the format below):\ni) What do you want to do next and why,\nii) what is your confidence about the correctness of the next step in form of [[#]] where # is number from 0 to 10, 0 meaning no confidence at all, and 10 meaning maximum confidence,\niii) which tool would be a good choice to execute this next step?"

    messages[-1]["content"][0]["text"] = f"{messages[-1]["content"][0]["text"]}{step1_prompt}"

    step1_response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        #tool_choice=None,
        #max_tokens=6666
    )
    completion_tokens += step1_response.usage.completion_tokens
    step1_tool_calls = [t.model_dump() for t in step1_response.choices[0].message.tool_calls]

    step1_response_text = step1_response.choices[0].message.content
    if step1_tool_calls:
        step1_response_text += f"\n\nHere are the tool calls I propose:\n{str(step1_tool_calls)}"

    print("--- step1 respose ---")
    print(step1_response.choices[0].finish_reason)
    print(step1_response.choices[0].message.content)
    cb_log["step1_prompt"] = step1_prompt
    cb_log["step1_response"] = step1_response_text
    cb_log["step1_tool_calls"] = step1_tool_calls
    cb_log["step1_finish_reason"] = step1_response.choices[0].finish_reason

    step2_prompt = "Can you execute the above step to generate the next interaction command for the environment"

    messages.append({"role": "assistant", "content": step1_response_text})
    messages.append({"role": "user", "content": step2_prompt})

    step2_response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        #tool_choice="required",
        #max_tokens=6666
    )
    if step2_response.choices[0].message.content:
        step2_response.choices[0].message.content = f"I will take the following approach:\n{step1_response.choices[0].message.content}\n\nMy next step:\n{step2_response.choices[0].message.content}"
    else:
        step2_response.choices[0].message.content = step1_response.choices[0].message.content
    completion_tokens += step2_response.usage.completion_tokens
    print("--- step2 respose ---")
    print(step2_response.choices[0].finish_reason)
    print(step2_response.choices[0].message.content)
    cb_log["step2_prompt"] = step2_prompt
    cb_log["step2_finish_reason"] = step2_response.choices[0].finish_reason
    
    step2_response.choices[0].cb_log = cb_log
    return step2_response, completion_tokens


def cepo_tool(messages: list, client: Any, model: str, cepo_config: CepoConfig, request_config: dict = None, request_id: str = None) -> tuple[str, int]:
    if 2 < cepo_config.tool_version <= 4:
        response, completion_tokens = globals()[f"cepo_tool_v{cepo_config.tool_version}"](messages, client, model, request_config)
    else:
        raise RuntimeError(f"Incorrect cepo tool version {cepo_config.tool_version}")
    
    return response, completion_tokens