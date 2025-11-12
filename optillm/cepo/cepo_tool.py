from typing import Literal, Any, Optional
from optillm.cepo.cepo import cepo, CepoConfig


def cepo_tool_v2(messages: list, client: Any, model: str, request_config: dict = None) -> tuple[str, int]:
    cb_log = {}
    cb_log["cepo_version"] = 2
    cb_log["cepo_version_description"] = ""
    completion_tokens = 0
    tools = request_config["tools"]

    step1_prompt = "\n\nLet's have an internal monologue before you decide on the next step interaction with the environment. To that end, can you state in natural language the following information: i) What do you want to do next and why, ii) which tool would be a good choice to execute this next step?"

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


def cepo_tool_v5(messages: list, client: Any, model: str, request_config: dict = None) -> tuple[str, int]:
    cb_log = {}
    cb_log["cepo_version"] = 5
    cb_log["cepo_version_description"] = "In step 2, additional ask to double check if the tool call meets all constraints"
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


def cepo_tool(messages: list, client: Any, model: str, cepo_config: CepoConfig, request_config: dict = None, request_id: str = None) -> tuple[str, int]:
    if 1 < cepo_config.tool_version <= 9:
        response, completion_tokens = globals()[f"cepo_tool_v{cepo_config.tool_version}"](messages, client, model, request_config)
    else:
        raise RuntimeError(f"Incorrect cepo tool version {cepo_config.tool_version}")
    
    return response, completion_tokens