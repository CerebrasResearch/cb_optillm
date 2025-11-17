from typing import Literal, Any, Optional, Tuple, List
from optillm.cepo.cepo import cepo, CepoConfig
import json, copy, re

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





##################################################################################
##################################################################################
##################################################################################
def _parse_router_json(text: str) -> dict:
    """
    Try to robustly extract a JSON object from the model output.
    Falls back to {} if nothing works.
    """
    if not text:
        return {}

    text = text.strip()
    # First, try direct JSON
    try:
        return json.loads(text)
    except Exception:
        pass

    # Fallback: grab the first {...} block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        candidate = match.group(0)
        try:
            return json.loads(candidate)
        except Exception:
            pass

    return {}


def _build_tool_inventory(tools: list) -> str:
    """
    Render the tools list into a human-readable inventory for the router prompt.
    Assumes OpenAI-style tool spec in request_config["tools"].
    """
    lines = ["Here is the list of tools you may choose from (names and descriptions):"]
    for t in tools:
        fn = t.get("function", {})
        name = fn.get("name", "<unknown>")
        desc = fn.get("description", "").strip()
        if desc:
            lines.append(f'- "{name}": {desc}')
        else:
            lines.append(f'- "{name}"')
    return "\n".join(lines)




# def cepo_tool_michael(messages: list, client: Any, model: str, request_config: dict = None) -> tuple[str, int]:
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


def _clean_monologue(text: str) -> str:
    if not text:
        return text

    # Remove explicit <tool_call>...</tool_call> blocks
    text = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL)

    # Remove any single-line pseudo-tags like <|tool_...|> or similar
    text = re.sub(r"<\|[^>]*tool[^>]*\|>", "", text)

    # Optionally nuke any line that starts with something that *looks* like a tag
    cleaned_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("<tool_call") or stripped.startswith("</tool_call"):
            continue
        if stripped.startswith("<|") and "tool" in stripped:
            continue
        cleaned_lines.append(line)
    text = "\n".join(cleaned_lines)

    return text.strip()


def cepo_tool_michael(
    messages: list,
    client: Any,
    model: str,
    request_config: dict = None,
) -> Tuple[Any, int]:
    cb_log = {}
    cb_log["cepo_version"] = 2
    cb_log["cepo_version_description"] = ""
    completion_tokens = 0
    tools = request_config["tools"]

    # ----- STEP 1: free-form internal monologue (NO tool calls) -----

    step1_plan_prompt = (
    "Let's plan in free form before you decide on the next step interaction "
    "with the environment. To that end, can you state your free-form thinking "
    "plan in natural language about what you want to do next and why?"
    )

    step1_constraints = """
IMPORTANT CONSTRAINTS FOR THIS STEP:
- In this step you must ONLY think in natural language.
- Do NOT write any JSON, XML, or code blocks.
- Do NOT write anything that looks like a tool call, such as:
  <tool_call>...</tool_call>, <|tool_...|>, or similar markup.
- Do NOT write arguments or schemas. Instead, just describe them in words.
  For example: "I will call TOOL_NAME to view file X with path Y".
- Your entire response must be plain sentences in natural language.

Again, this is only internal thinking. You are NOT actually calling tools yet.
""".strip()



    base_messages = copy.deepcopy(messages)

    # Step 1: planning-only call (no tools)
    messages_step1 = base_messages + [
        {
            "role": "user",
            "content": f"{step1_plan_prompt}\n\n{step1_constraints}",
        }
    ]

    # IMPORTANT: no tools here → pure free-form thinking, no tool_calls
    step1_response = client.chat.completions.create(
        model=model,
        messages=messages_step1,
        temperature=0.85,
        # no tools argument → model can't emit tool_calls in this step
    )
    completion_tokens += step1_response.usage.completion_tokens

    print("--- step1 response ---")
    print(step1_response.choices[0].finish_reason)
    print(step1_response.choices[0].message.content)

    step1_raw = step1_response.choices[0].message.content or ""
    step1_clean = _clean_monologue(step1_raw)

    cb_log["step1_plan_prompt"] = step1_plan_prompt
    cb_log["step1_constraints"] = step1_constraints
    cb_log["step1_response_raw"] = step1_raw
    cb_log["step1_response_clean"] = step1_clean
    cb_log["step1_finish_reason"] = step1_response.choices[0].finish_reason


    # ----- STEP 2: execute the plan with a real tool call -----

    step2_prompt = (
        "Can you execute the above plan to generate the next appropriate tool and interaction command for "
        "the environment? First revisit the plan you had, "
        "then make sure the tool you are going to pick now is consistent with "
        "what you produced above in the free-form thinking."
    )

    # For step 2, we add the step 1 monologue as an assistant message,
    # then ask the model to actually choose & call a tool.
    messages_step2 = base_messages + [
        {"role": "assistant", "content": step1_clean},
        {"role": "user", "content": step2_prompt},
    ]


    step2_response = client.chat.completions.create(
        model=model,
        messages=messages_step2,
        tools=tools,
        temperature=0.5,
        tool_choice="required",
        # optionally: tool_choice="required"
        # max_tokens=...
    )
    completion_tokens += step2_response.usage.completion_tokens

    print("--- step2 response ---")
    print(step2_response.choices[0].finish_reason)
    print(step2_response.choices[0].message.content)

    cb_log["step2_prompt"] = step2_prompt
    cb_log["step2_finish_reason"] = step2_response.choices[0].finish_reason

    # Attach the monologue to the natural language content (for transparency),
    # but this does NOT affect the tool_calls the environment will parse.
    if step2_response.choices[0].message.content:
        step2_response.choices[0].message.content = (
            "I will take the following approach:\n"
            f"{step1_response.choices[0].message.content}\n\n"
            "My next step:\n"
            f"{step2_response.choices[0].message.content}"
        )
    else:
        # If there's no textual content, at least expose the monologue
        step2_response.choices[0].message.content = (
            step1_response.choices[0].message.content
        )

    # ----- POST-STEP-2: detect code-edit tool call and branch into best-of-N + reflection -----

    # Extract tool calls from step 2 (if any)
    step2_tool_calls = step2_response.choices[0].message.tool_calls or []
    cb_log["step2_tool_calls"] = [
        t.model_dump() for t in step2_tool_calls
    ]

    # Default: return step2_response as-is (non-code-edit or no tool_calls)
    final_response = step2_response

    # If there is at least one tool call, check if it's a non-view/non-create str_replace_editor
    if step2_tool_calls:
        first_tc = step2_tool_calls[0]
        fn_name = first_tc.function.name

        if fn_name == "str_replace_editor":
            # arguments may be a JSON string; parse it
            try:
                step2_argument = (
                    json.loads(first_tc.function.arguments)
                    if isinstance(first_tc.function.arguments, str)
                    else first_tc.function.arguments
                )
            except Exception:
                step2_argument = {}

            command = step2_argument.get("command")

            # Treat as *code-edit* only if it's not a pure view/create request
            if command not in ("view", "create"):
                # ----- CODE-EDIT PATH: best-of-N + self-reflection -----

                # Messages for code-generation runs:
                # we include the free-form monologue so the model sees its own plan.
                base_code_gen_messages = base_messages + [
                    {"role": "assistant", "content": step1_clean}
                ]
                code_gen_messages_for_reflection = base_code_gen_messages  # same in this setup

                code_gen_responses = []
                for _ in range(2):  # N=2; bump to 3 if you want more diversity
                    code_gen_resp, code_gen_tokens, code_gen_prompt = single_code_edition(
                        base_code_gen_messages, client, model, tools
                    )
                    code_gen_responses.append(code_gen_resp)
                    completion_tokens += code_gen_tokens

                reflection_response, reflection_completion_tokens, reflection_prompt = self_reflection(
                    code_gen_messages_for_reflection,
                    client,
                    model,
                    code_gen_responses,
                    tools,
                )
                completion_tokens += reflection_completion_tokens

                print("--- reflection (step2 override) response ---")
                print(reflection_response.choices[0].finish_reason)
                print(reflection_response.choices[0].message.content)

                # Logging for the code-edit path
                cb_log["step2_code_edit"] = {
                    "trigger_tool": fn_name,
                    "trigger_args": step2_argument,
                    "code_gen_prompt": code_gen_prompt,
                    "code_gen_responses": [
                        {
                            "content": r.choices[0].message.content,
                            "tool_calls": [
                                t.model_dump()
                                for t in (r.choices[0].message.tool_calls or [])
                            ],
                        }
                        for r in code_gen_responses
                    ],
                    "reflection_prompt": reflection_prompt,
                }
                cb_log["step2_reflection_finish_reason"] = (
                    reflection_response.choices[0].finish_reason
                )

                final_response = reflection_response

    # Attach cb_log to the final response choice
    final_response.choices[0].cb_log = cb_log

    # Notice that we are returning the full response object including tool calls back to OpenHands
    return final_response, completion_tokens



def single_code_edition(base_messages: list, client: Any, model: str, tools: list):
    code_gen_prompt = (
        "Before diving into editing the code, please first write out your plan for this code edit in natural language."
        "Revisit the OpenHands requirements at the start of conversation, and make sure your plan follows it."
        "In your edit, please do double check all import are correct, and all function call arguments are legit, and all functions are implemented in the correct place."
        "Be extra careful! Never make changes to ANY non-python configuration files and existing test files! The enviroment is already perfect and should not require any new change. Again, please avoid editing configuration, enviornoment-related, and existing test files!"
        "Always double check whether multiple existing non-test files in the repository need to be editted."
    )
    messages = base_messages + [{"role": "user", "content": code_gen_prompt}]
    codegen_response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        temperature=0.9,
        tool_choice="required",
    )
    return codegen_response, codegen_response.usage.completion_tokens, code_gen_prompt


def self_reflection(
    messages: List[dict],
    client: Any,
    model: str,
    code_responses: List[Any],
    tools: List, 
) -> Tuple[Any, int]:

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
        temperature=0.3,
        tool_choice="required",
    )

    return reflection_response, reflection_response.usage.completion_tokens, reflection_prompt




def cepo_tool(messages: list, client: Any, model: str, cepo_config: CepoConfig, request_config: dict = None, request_id: str = None) -> tuple[str, int]:
    # if 1 < cepo_config.tool_version <= 9:
    #     response, completion_tokens = globals()[f"cepo_tool_v{cepo_config.tool_version}"](messages, client, model, request_config)
    # else:
    #     raise RuntimeError(f"Incorrect cepo tool version {cepo_config.tool_version}")
    response, completion_tokens = cepo_tool_michael(messages, client, model, request_config)
    
    return response, completion_tokens