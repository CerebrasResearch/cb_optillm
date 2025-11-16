from typing import Literal, Any, Optional, Tuple, List
from optillm.cepo.cepo import cepo, CepoConfig
import json, copy

def cepo_tool_v2(messages: list, client: Any, model: str, request_config: dict = None) -> tuple[str, int]:
    cb_log = {}
    cb_log["cepo_version"] = 2
    cb_log["cepo_version_description"] = ""
    completion_tokens = 0
    tools = request_config["tools"]

    step1_prompt = "\n\nLet's have an internal monologue before you decide on the next step interaction with the environment. To that end, can you state in natural language the following information: i) What do you want to do next and why, ii) which tool would be a good choice to execute this next step?"

    messages[-1]["content"][0]["text"] = f"{messages[-1]['content'][0]['text']}{step1_prompt}"

    step1_response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        #tool_choice=None,
        #max_tokens=6666
    )
    completion_tokens += step1_response.usage.completion_tokens
    print("--- step1 response ---")
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


def cepo_tool_v3(messages: list, client: Any, model: str, request_config: dict = None) -> tuple[str, int]:
    cb_log = {}
    cb_log["cepo_version"] = 3
    cb_log["cepo_version_description"] = "Requesting the model to evaluate its confidence of the next step"
    completion_tokens = 0
    tools = request_config["tools"]

    step1_prompt = "\n\nLet's have an internal monologue before you decide on the next step interaction with the environment. To that end, can you state in natural language the following information (follow the format below):\ni) What do you want to do next and why,\nii) what is your confidence about the correctness of the next step in form of [[#]] where # is number from 0 to 10, 0 meaning no confidence at all, and 10 meaning maximum confidence,\niii) which tool would be a good choice to execute this next step?"

    messages[-1]["content"][0]["text"] = f"{messages[-1]['content'][0]['text']}{step1_prompt}"

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

    messages[-1]["content"][0]["text"] = f"{messages[-1]['content'][0]['text']}{step1_prompt}"

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


def cepo_tool_v5(messages: list, client: Any, model: str, request_config: dict = None) -> tuple[str, int]:
    cb_log = {}
    cb_log["cepo_version"] = 5
    cb_log["cepo_version_description"] = "In step 2, additional ask to double check if the tool call meets all constraints"
    completion_tokens = 0
    tools = request_config["tools"]

    step1_prompt = "\n\nLet's have an internal monologue before you decide on the next step interaction with the environment. To that end, can you state in natural language the following information (follow the format below):\ni) What do you want to do next and why,\nii) what is your confidence about the correctness of the next step in form of [[#]] where # is number from 0 to 10, 0 meaning no confidence at all, and 10 meaning maximum confidence,\niii) which tool would be a good choice to execute this next step?"

    messages[-1]["content"][0]["text"] = f"{messages[-1]['content'][0]['text']}{step1_prompt}"

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

    step2_prompt = "Can you execute the above step to generate the next interaction command for the environment? Please double check that the tool invocation actually follows all the constraints in the tool description and is correct."

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


def cepo_tool_v6(messages: list, client: Any, model: str, request_config: dict = None) -> tuple[str, int]:
    #TODO postponed for later, moving on to v7 and v8
    raise NotImplemented()
    cb_log = {}
    cb_log["cepo_version"] = 6
    cb_log["cepo_version_description"] = "Run step 2 multiple times"
    completion_tokens = 0
    tools = request_config["tools"]

    step1_prompt = "\n\nLet's have an internal monologue before you decide on the next step interaction with the environment. To that end, can you state in natural language the following information (follow the format below):\ni) What do you want to do next and why,\nii) what is your confidence about the correctness of the next step in form of [[#]] where # is number from 0 to 10, 0 meaning no confidence at all, and 10 meaning maximum confidence,\niii) which tool would be a good choice to execute this next step?"

    messages[-1]["content"][0]["text"] = f"{messages[-1]['content'][0]['text']}{step1_prompt}"

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

    step2_prompt = "Can you execute the above step to generate the next interaction command for the environment? Please double check that the tool invocation actually follows all the constraints in the tool description and is correct."
    messages.append({"role": "assistant", "content": step1_response.choices[0].message.content})
    messages.append({"role": "user", "content": step2_prompt})
    cb_log["step2_prompt"] = step2_prompt

    n = 3
    step2_response = []
    for i in range(n):
        step2_response_i = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            #tool_choice="required",
            #max_tokens=6666
        )
        completion_tokens += step2_response_i.usage.completion_tokens
        print(f"--- step2 respose {i+1}/{n} ---")
        print(step2_response_i.choices[0].finish_reason)
        print(step2_response_i.choices[0].message.content)    
        cb_log[f"step2_response_{i}"] = step2_response_i.choices[0].message.content
        cb_log[f"step2_finish_reason_{i}"] = step2_response_i.choices[0].finish_reason
        step2_response.append(step2_response_i)

    plans_message = ""
    for i, plan in enumerate(step2_response):
        plans_message += f"Response {i + 1}:\n{plan}\n\n"
    plans_message = plans_message.rstrip()
    plans_message = f"Here are {n} answer proposals:\n\n{plans_message}"

    content = f"Can you review your last {len(step1_response)} responses and identify any inconsistency between them. After that, can you address "\
              f"it and present the final answer."

    
    messages = [{"role": "assistant", "content": plans_message}, {"role": "user", "content": content}]
    
    provider_request = {
                "model": model,
                "messages": messages,
                "max_tokens": cepo_config.planning_max_tokens_step1,
                "temperature": cepo_config.planning_temperature_step1,
                "top_p": 1.0
                }
    
    response, finish_reason, completion_tokens_ = llm_call_reason_effort_fallback(
                client=client,
                provider_request=provider_request,
                reasoning_effort_levels=["high", "medium"],
                cepo_config=cepo_config
            )
    completion_tokens += completion_tokens_
    
    step2_response_i.choices[0].cb_log = cb_log
    return step2_response_i, completion_tokens


def cepo_tool_v7(messages: list, client: Any, model: str, request_config: dict = None) -> tuple[str, int]:
    cb_log = {}
    cb_log["cepo_version"] = 7
    cb_log["cepo_version_description"] = "New prompt for step 2"
    completion_tokens = 0
    tools = request_config["tools"]

    step1_prompt = "\n\nLet's have an internal monologue before you decide on the next step interaction with the environment. To that end, can you state in natural language the following information (follow the format below):\ni) What do you want to do next and why,\nii) what is your confidence about the correctness of the next step in form of [[#]] where # is number from 0 to 10, 0 meaning no confidence at all, and 10 meaning maximum confidence,\niii) which tool would be a good choice to execute this next step?"

    messages[-1]["content"][0]["text"] = f"{messages[-1]['content'][0]['text']}{step1_prompt}"

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

    step2_prompt = "Your only task in this turn is to turn the internal monologue you produced above into a single JSON tool call that will be sent to the coding sandbox. Please use the tool identified in the internal monologue and create a detailed JSON tool object that specifies all its arguments.\n\nImportant points to keep in mind:\n- **Nothing from Step 1 has been run yet.** Treat the monologue as pure *information* that still needs to be turned into a command.\n- The response you give must be **exactly one JSON object** and **nothing else** (no extra sentences, no markdown fences, no code blocks)."

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


def cepo_tool_v8(messages: list, client: Any, model: str, request_config: dict = None) -> tuple[str, int]:
    cb_log = {}
    cb_log["cepo_version"] = 8
    cb_log["cepo_version_description"] = "New prompt for step 1"
    completion_tokens = 0
    tools = request_config["tools"]

    step1_prompt = "Before your next interaction with the environment, do a brief *internal monologue* that contains **exactly** the three items below, in plain English, and NOTHING else:\n1. **Goal for the next interaction** - state, in one sentence, what you want to accomplish and why it moves the overall task forward.\n2. **Chosen tool** - give the exact tool name from the catalog (e.g., `run_python`, `search_web`, `read_file`, `write_file`, `list_dir`, `git_clone`, etc.) and a short justification (one clause).\n3. **Natural call description** - write in natural language the information you will pass to that tool. Do **not** create a JSON object, code fences, or any markup."

    messages[-1]["content"][0]["text"] = f"{messages[-1]['content'][0]['text']}{step1_prompt}"

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

    step2_prompt = "Your only task in this turn is to turn the internal monologue you produced above into a tool call that will be sent to the coding sandbox. Please use the tool identified in the internal monologue and create a detailed tool call that specifies all its arguments.\n\nImportant point to keep in mind is **Nothing from Step 1 has been run yet.** Treat the monologue as pure *information* that still needs to be turned into a command."

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


def cepo_tool_v9(messages: list, client: Any, model: str, request_config: dict = None) -> tuple[str, int]:
    cb_log = {}
    cb_log["cepo_version"] = 9
    cb_log["cepo_version_description"] = "Modified step 2 prompt to make it more robust in generating tool calls"
    completion_tokens = 0
    tools = request_config["tools"]

    step1_prompt = "\n\nLet's have an internal monologue before you decide on the next step interaction with the environment. To that end, can you state in natural language the following information (follow the format below):\ni) What do you want to do next and why,\nii) what is your confidence about the correctness of the next step in form of [[#]] where # is number from 0 to 10, 0 meaning no confidence at all, and 10 meaning maximum confidence,\niii) which tool would be a good choice to execute this next step?"

    messages[-1]["content"][0]["text"] = f"{messages[-1]['content'][0]['text']}{step1_prompt}"

    step1_response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        #tool_choice=None,
        #max_tokens=6666
    )
    completion_tokens += step1_response.usage.completion_tokens
    print("--- step1 response ---")
    print(step1_response.choices[0].finish_reason)
    print(step1_response.choices[0].message.content)
    cb_log["step1_prompt"] = step1_prompt
    cb_log["step1_response"] = step1_response.choices[0].message.content
    cb_log["step1_tool_calls"] = [t.model_dump() for t in step1_response.choices[0].message.tool_calls]
    cb_log["step1_finish_reason"] = step1_response.choices[0].finish_reason

    step2_prompt = "Your only task in this turn is to turn the internal monologue you produced above into a tool call that will be sent to the coding sandbox. Please use the tool identified in the internal monologue and create a detailed tool call that specifies all its arguments.\n\nImportant point to keep in mind is **Nothing from Step 1 has been run yet.** Treat the monologue as pure *information* that still needs to be turned into a command."

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



def cepo_tool_michael(messages: list, client: Any, model: str, request_config: dict = None) -> tuple[str, int]:
    cb_log = {}
    cb_log["cepo_version"] = 2
    cb_log["cepo_version_description"] = ""
    completion_tokens = 0
    tools = request_config["tools"]

    step1_prompt = "\n\nLet's have an internal monologue before you decide on the next step interaction with the environment. To that end, can you state in natural language the following information: i) What do you want to do next and why, ii) which tool would be a good choice to execute this next step?"

    messages[-1]["content"][0]["text"] = f"{messages[-1]['content'][0]['text']}{step1_prompt}"

    step1_response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        #tool_choice=None,
        #max_tokens=6666
    )
    completion_tokens += step1_response.usage.completion_tokens
    print("--- step1 response ---")
    print(step1_response.choices[0].finish_reason)
    print(step1_response.choices[0].message.content)
    step1_tool_calls = [t.model_dump() for t in step1_response.choices[0].message.tool_calls]
    cb_log["step1_prompt"] = step1_prompt
    cb_log["step1_response"] = step1_response.choices[0].message.content
    cb_log["step1_tool_calls"] = step1_tool_calls
    cb_log["step1_finish_reason"] = step1_response.choices[0].finish_reason

    # Case 1: step 1 does imply a code edit operation
    if step1_tool_calls[0]["function"]["name"] == "str_replace_editor":
        # e.g., {'command': 'view', 'path': '/workspace', 'security_risk': 'LOW'}
        step1_argument = json.loads(step1_tool_calls[0]["function"]["arguments"])
        if step1_argument["command"] != "view" and step1_argument["command"] != "create":
            code_gen_messages = messages + [{"role": "assistant", "content": step1_response.choices[0].message.content}]
            code_gen_responses = []
            base_code_gen_messages = messages + [
                        {"role": "assistant", "content": step1_response.choices[0].message.content}
                    ]
            for _ in range(2):
                code_gen_resp, code_gen_tokens, code_gen_prompt = single_code_edition(
                    base_code_gen_messages, client, model, tools
                )
                code_gen_responses.append(code_gen_resp)
                completion_tokens += code_gen_tokens

            reflection_response, reflection_completion_tokens, reflection_prompt = self_reflection(code_gen_messages, client, model, code_gen_responses, tools)
            step2_response = reflection_response
            completion_tokens += reflection_completion_tokens
            print("--- step2 response ---")
            print(step2_response.choices[0].finish_reason)
            print(step2_response.choices[0].message.content)
            cb_log["step2_prompt"] = {
                "code_gen_prompt": code_gen_prompt,
                "code_gen_responses": [
                    {
                        "content": r.choices[0].message.content,
                        "tool_calls": [
                            t.model_dump() for t in (r.choices[0].message.tool_calls or [])
                        ],
                    }
                    for r in code_gen_responses
                ],
                "reflection_prompt": reflection_prompt,
            }

            cb_log["step2_finish_reason"] = step2_response.choices[0].finish_reason
            step2_response.choices[0].cb_log = cb_log
            # Notice that we are returning the full response object including tool calls back to openhands
            return step2_response, completion_tokens 

    # Case 2: step 1 doesn't imply a code edit operation

    step2_prompt = (
        "Can you execute the above step to generate the next interaction command for the environment?"
        "First revisit the tool you picked above in the monologue, then Make sure the tool you are going to pick now is exactly the same as what you produced above in the monologue."
    )
    
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
    print("--- step2 response ---")
    print(step2_response.choices[0].finish_reason)
    print(step2_response.choices[0].message.content)
    cb_log["step2_prompt"] = step2_prompt
    cb_log["step2_finish_reason"] = step2_response.choices[0].finish_reason
    
    step2_response.choices[0].cb_log = cb_log
    # Notice that we are returning the full response object including tool calls back to openhands
    return step2_response, completion_tokens


def single_code_edition(base_messages: list, client: Any, model: str, tools: list):
    code_gen_prompt = (
        "Great! Now you are ready to edit some code."
        # "Before diving into editting, please first write out your plan for this code edit in natural language."
        # "Revisit the OpenHands requirements at the start of conversation, and make sure your plan follows it."
        # "In your edit, please do double check all import are correct, and all function call arguments are legit, and all functions are implemented in the correct place."
        "Your code should be concise and not over-complex; do not over-engineer the problems!"
        "Do not make changes to configuration and existing test files."
        "Always Double check whether multiple existing non-test files in the repository need to be editted."
    )
    messages = base_messages + [{"role": "user", "content": code_gen_prompt}]
    codegen_response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        temperature=0.8,
    )
    return codegen_response, codegen_response.usage.completion_tokens, code_gen_prompt


def self_reflection(
    messages: List[dict],
    client: Any,
    model: str,
    code_responses: List[Any],
    tools: List, 
) -> Tuple[Any, int]:
    """
    Take multiple code-generation responses (with tool calls) and ask the model
    to reflect on them and synthesize a single improved tool call.

    Just return the raw OpenAI response object and OpenHands can take from there to do tool parsing etc
    """

    # --- 1. Extract tool calls from each candidate ---
    candidates = []
    for i, resp in enumerate(code_responses):
        # Adjust if your client object has a different shape
        choice_msg = resp.choices[0].message

        candidate_info = {"index": i + 1}
        tool_calls = getattr(choice_msg, "tool_calls", None)

        if tool_calls:
            tc_list = []
            for tc in tool_calls:
                # tc.function.arguments is usually a JSON string
                try:
                    args = (
                        json.loads(tc.function.arguments)
                        if isinstance(tc.function.arguments, str)
                        else tc.function.arguments
                    )
                except Exception:
                    # Fallback: keep raw string if it fails to parse
                    args = tc.function.arguments

                tc_list.append(
                    {
                        "function_name": tc.function.name,
                        "arguments": args,
                    }
                )
            candidate_info["tool_calls"] = tc_list
        else:
            # If there were no tool calls, keep content for reference
            candidate_info["content"] = choice_msg.content

        candidates.append(candidate_info)

    # --- 2. Build a text block listing the candidates' tool calls ---
    candidate_blocks = []
    for cand in candidates:
        idx = cand["index"]
        if "tool_calls" in cand:
            block_lines = [f"Candidate {idx}:"]
            for j, tc in enumerate(cand["tool_calls"], start=1):
                block_lines.append(f"  Tool call {j}:")
                block_lines.append(f"    function_name: {tc['function_name']}")
                block_lines.append("    arguments:")
                block_lines.append(
                    "      " + json.dumps(tc["arguments"], indent=2).replace("\n", "\n      ")
                )
            candidate_blocks.append("\n".join(block_lines))
        else:
            candidate_blocks.append(
                f"Candidate {idx} (no tool calls, only content):\n  {cand['content']}"
            )

    candidates_text = "\n\n".join(candidate_blocks)

    # --- 3. Reflection prompt asking for ONE final tool call as JSON ---
    reflection_prompt = f"""
    You previously generated {len(candidates)} different tool-based code-edit attempts
    for solving the same software engineering problem. Each attempt consists of at least one tool call.

    Now I want you to:
    1. Compare these candidates and identify inconsistencies, mistakes, or missing cases in their code-edit plans and arguments.
    2. Decide on a single improved code-edit plan, possibly combining ideas from multiple candidates.
    3. Synthesize ONE final tool call that best fixes the bug.

    Here are the previous candidates:

    {candidates_text}
    """.strip()

    # We avoid mutating the original messages list
    new_messages = copy.deepcopy(messages)
    new_messages.append({"role": "user", "content": reflection_prompt})

    # --- 4. Call the model (no tools here; we want JSON text back) ---
    reflection_response = client.chat.completions.create(
        model=model,
        messages=new_messages,
        tools=tools,
    )

    return reflection_response, reflection_response.usage.completion_tokens, reflection_prompt




def cepo_tool(messages: list, client: Any, model: str, cepo_config: CepoConfig, request_config: dict = None, request_id: str = None) -> tuple[str, int]:
    # if 1 < cepo_config.tool_version <= 9:
    #     response, completion_tokens = globals()[f"cepo_tool_v{cepo_config.tool_version}"](messages, client, model, request_config)
    # else:
    #     raise RuntimeError(f"Incorrect cepo tool version {cepo_config.tool_version}")
    response, completion_tokens = cepo_tool_michael(messages, client, model, request_config)
    
    return response, completion_tokens