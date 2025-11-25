from typing import Literal, Any, Optional, Tuple, List
from optillm.cepo.cepo import cepo, CepoConfig
import json, copy, re
from openai import OpenAI


gpt_oss_120b_client = OpenAI(api_key="serving-on-vllm",
                            base_url="http://localhost:8190/v1",
                            max_retries=0, 
                            timeout=None)


def cepo_tool(messages: list, client: Any, model: str, cepo_config: CepoConfig, request_config: dict = None, request_id: str = None):
    # if 1 < cepo_config.tool_version <= 9:
    #     response, completion_tokens = globals()[f"cepo_tool_v{cepo_config.tool_version}"](messages, client, model, request_config)
    # else:
    #     raise RuntimeError(f"Incorrect cepo tool version {cepo_config.tool_version}")
    response, completion_tokens = cepo_tool_michael(messages, client, model, request_config)

    # response, completion_tokens = cepo_tool_v2(messages, client, model, request_config)
    
    return response, completion_tokens

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



##################################################################################
##################################################################################
##################################################################################


def call_with_required_tool_retry(
    *,
    client: Any,
    model: str,
    tools: list,
    base_messages: list,
    temperature: float = 0.5,
    max_retries: int = 2,
    retry_temperature: float | None = None,
) -> Tuple[Any, int]:
    total_completion_tokens = 0
    messages = copy.deepcopy(base_messages)
    temp = temperature
    if retry_temperature is None:
        retry_temperature = min(temperature, 0.3)

    last_response = None

    # decide if this model supports reasoning_effort
    supports_reasoning_effort = "gpt-oss" in model  # adjust if you have a different naming scheme

    for attempt in range(max_retries + 1):
        extra_kwargs = {}
        if supports_reasoning_effort:
            extra_kwargs["reasoning_effort"] = "high"
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                temperature=temp,
                tool_choice="required",
                **extra_kwargs,
            )
        except:
            breakpoint()
        last_response = response
        total_completion_tokens += response.usage.completion_tokens

        msg = response.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None) or []

        if tool_calls:
            return response, total_completion_tokens

        if attempt < max_retries:
            strict_retry_prompt = """
Your previous response did not contain any tool calls, which is invalid.

In this step you MUST:
- Respond using one tool call from the provided tool list.
- NOT answer with plain text explanation alone.
- NOT wrap the tool call in JSON arrays or other text.
Simply call the correct tool with appropriate arguments.

If you do not produce a tool call, you are failing the task.
""".strip()

            messages = messages + [
                {"role": "assistant", "content": msg.content or ""},
                {"role": "user", "content": strict_retry_prompt},
            ]
            temp = retry_temperature
        else:
            return last_response, total_completion_tokens

    return last_response, total_completion_tokens





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


def normalize_messages_for_chat(messages: list) -> list:
    """
    Ensure every message has content as a plain string.

    - If content is already a string, leave it.
    - If content is a list of {type: "text", text: "..."} blocks, join them.
    - Otherwise, cast to string (best-effort fallback).
    """
    normalized = []
    for m in messages:
        m = copy.deepcopy(m)
        content = m.get("content")

        # Already a simple string
        if isinstance(content, str):
            normalized.append(m)
            continue

        # List of blocks (OpenAI/Together / OpenHands style)
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
                # ignore non-text blocks (images, etc.) for now
            m["content"] = "\n".join(parts)
            normalized.append(m)
            continue

        # Fallback: just stringify whatever it is
        if content is None:
            m["content"] = ""
        else:
            m["content"] = str(content)

        normalized.append(m)

    return normalized



def cepo_tool_michael(
    messages: list,
    client: Any,
    model: str,
    request_config: dict = None,
):
    cb_log = {}
    cb_log["cepo_version_description"] = ""
    completion_tokens = 0
    tools = request_config["tools"]


    # Normalize all incoming messages once at the entry point, this is required for gpt-oss and doesn't conflict with together 480B
    base_messages = normalize_messages_for_chat(copy.deepcopy(messages))

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
- Your entire response must be plain sentences in natural language.
Again, this is only internal thinking. You are NOT actually calling tools yet.

Test-related planning guidelines:
- You ARE allowed to create or modify tests, but you must be very concise.
- Only add or change tests when they directly help verify the specific bug or requirement in this SWE task.
- Before planning any new tests, check that:
  (1) The behavior you test truly matches what the problem statement requires.
  (2) Your test structure, naming, and assertions follow the style and format of existing tests in this repository.
- Do NOT create large, speculative new test suites or many redundant test cases.

""".strip()

    messages_step1 = base_messages + [
        {
            "role": "user",
            "content": f"{step1_plan_prompt}\n\n{step1_constraints}",
        }
    ]

    step1_response = client.chat.completions.create(
        model=model,
        messages=messages_step1,
        temperature=0.7,
    )
    completion_tokens += step1_response.usage.completion_tokens

    # print("--- step1 response ---")
    # print(step1_response.choices[0].finish_reason)
    # print(step1_response.choices[0].message.content)

    step1_raw = step1_response.choices[0].message.content or ""
    step1_clean = _clean_monologue(step1_raw)

    cb_log["step1_plan_prompt"] = step1_plan_prompt
    cb_log["step1_constraints"] = step1_constraints
    cb_log["step1_response_raw"] = step1_raw
    cb_log["step1_response_clean"] = step1_clean
    cb_log["step1_finish_reason"] = step1_response.choices[0].finish_reason

    # ----- STEP 2: execute the plan with a real tool call -----

    step2_prompt = (
        "Now execute the action you planned above and generate appropriate tools "
        "Make sure the tool you are going to pick now is consistent with "
        "what you produced above in the free-form thinking."
    )

    messages_step2 = base_messages + [
        {"role": "user", "content": step1_plan_prompt},
        {"role": "assistant", "content": step1_clean},
        {"role": "user", "content": step2_prompt},
    ]

    # USE WRAPPER HERE
    step2_response, step2_comp_tokens = call_with_required_tool_retry(
        client=client,
        model=model,
        tools=tools,
        base_messages=messages_step2,
        temperature=0.5,
        max_retries=1,
        retry_temperature=0.2,
    )
    completion_tokens += step2_comp_tokens

    # print("--- step2 response ---")
    # print(step2_response.choices[0].finish_reason)
    # print(step2_response.choices[0].message.content)

    cb_log["step2_prompt"] = step2_prompt
    cb_log["step2_finish_reason"] = step2_response.choices[0].finish_reason

    # Attach monologue to natural language content (for transparency)
    if step2_response.choices[0].message.content:
        step2_response.choices[0].message.content = (
            "I will take the following approach:\n"
            f"{step1_raw}\n\n"
            "My next step:\n"
            f"{step2_response.choices[0].message.content}"
        )
    else:
        step2_response.choices[0].message.content = step1_raw

    # ----- POST-STEP-2: detect code-edit tool call and branch into best-of-N + reflection -----

    step2_tool_calls = getattr(step2_response.choices[0].message, "tool_calls", None) or []
    cb_log["step2_tool_calls"] = [t.model_dump() for t in step2_tool_calls]

    final_response = step2_response

    if step2_tool_calls:
        first_tc = step2_tool_calls[0]
        fn_name = first_tc.function.name

        if fn_name == "str_replace_editor":
            try:
                step2_argument = (
                    json.loads(first_tc.function.arguments)
                    if isinstance(first_tc.function.arguments, str)
                    else first_tc.function.arguments
                )
            except Exception:
                step2_argument = {}

            command = step2_argument.get("command")

            if command not in ("view",):
                # ----- CODE-EDIT PATH: best-of-N + self-reflection -----

                base_code_gen_messages = base_messages + [
                    {"role": "user", "content": step1_plan_prompt},
                    {"role": "assistant", "content": step1_clean}
                ]
                code_gen_messages_for_reflection = base_code_gen_messages

                code_gen_responses = []
                code_gen_clients = [
                    (client, model),                           # e.g. qwen
                    (gpt_oss_120b_client, "openai/gpt-oss-120b"),  # gpt-oss model name
                ]

                for client_to_use, model_name in code_gen_clients:
                    code_gen_resp, code_gen_tokens, code_gen_prompt = single_code_edition(
                        base_code_gen_messages, client_to_use, model_name, tools
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

                cb_log["step2_code_edit"] = {
                    "trigger_tool": fn_name,
                    "trigger_args": step2_argument,
                    "code_gen_prompt": code_gen_prompt,
                    "code_gen_responses": [
                        {
                            "content": r.choices[0].message.content,
                            "tool_calls": [
                                t.model_dump()
                                for t in (getattr(r.choices[0].message, "tool_calls", None) or [])
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

    final_response.choices[0].cb_log = cb_log
    return final_response, completion_tokens



def single_code_edition(base_messages: list, client: Any, model: str, tools: list):
    code_gen_prompt = (
    "Before diving into editing the code, please first write out your plan for this code edit in natural language. "
    "Revisit the OpenHands requirements at the start of the conversation and make sure your plan follows them. "
    "In your edit, double-check that all imports are correct and all function calls use valid arguments. "
    "IMPORTANT: Prefer the smallest, most localized change that can solve the issue.\n\n"
    "Test-related editing guidelines:\n"
    "- You ARE allowed to create or modify tests, but only when it directly helps verify the specific bug or requirement. "
    "- Keep any new or modified tests very concise: add at most a small number of focused test cases that target the bug. "
    "- Make sure your tests:\n"
    "  (1) Match the intended behavior in the original problem statement.\n"
    "  (2) Follow the existing test style, naming patterns, and assertion structure in this repository.\n"
    "- Do NOT create large, speculative new test suites or many redundant tests.\n\n"
    "Config and environment guidelines:\n"
    "- You should never edit configuration or environment-related files or code, no matter what.\n"
)


    messages = base_messages + [{"role": "user", "content": code_gen_prompt}]

    codegen_response, completion_tokens = call_with_required_tool_retry(
        client=client,
        model=model,
        tools=tools,
        base_messages=messages,
        temperature=0.7,
        max_retries=1,
        retry_temperature=0.5,
    )

    return codegen_response, completion_tokens, code_gen_prompt


def self_reflection(
    messages: List[dict],
    client: Any,
    model: str,
    code_responses: List[Any],
    tools: List, 
) -> Tuple[Any, int, str]:

    candidates = []
    for i, resp in enumerate(code_responses):
        choice_msg = resp.choices[0].message
        candidate_info = {"index": i + 1}
        tool_calls = getattr(choice_msg, "tool_calls", None)

        if tool_calls:
            tc_list = []
            for tc in tool_calls:
                try:
                    args = (
                        json.loads(tc.function.arguments)
                        if isinstance(tc.function.arguments, str)
                        else tc.function.arguments
                    )
                except Exception:
                    args = tc.function.arguments

                tc_list.append(
                    {
                        "function_name": tc.function.name,
                        "arguments": args,
                    }
                )
            candidate_info["tool_calls"] = tc_list
        else:
            candidate_info["content"] = choice_msg.content

        candidates.append(candidate_info)

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

    reflection_prompt = f"""
You previously generated {len(candidates)} different tool-based code-edit attempts
for solving the same software engineering problem. Each attempt consists of at least one tool call.

Now I want you to:
1. Compare these candidates and identify inconsistencies, mistakes, or missing cases in their code-edit plans and arguments.
2. Decide on a single improved code-edit plan, possibly combining ideas from multiple candidates.
3. Synthesize ONE final tool call that best fixes the bug.

When comparing and choosing between candidates, follow these principles:

- Prefer the smallest, most localized code change that can plausibly fix the failing tests.
- Only create or modify tests when it directly helps verify the specific bug or requirement in this SWE task.
- Penalize candidates that add many new tests or large test files without clear justification.
- Prefer candidates whose tests (if any are added or modified):
  (1) Match the intended behavior described in the original problem statement.
  (2) Follow the existing test style, naming patterns, and assertion structure in this repository.
- Avoid over-engineering: do NOT select candidates that introduce large, speculative new test suites, broad refactors, or unrelated changes.

Your final response MUST be expressed as a single tool call from the available tools.
Do NOT answer with plain text only.

Here are the previous candidates:

{candidates_text}
""".strip()


    new_messages = copy.deepcopy(messages)
    new_messages.append({"role": "user", "content": reflection_prompt})

    reflection_response, completion_tokens = call_with_required_tool_retry(
        client=client,
        model=model,
        tools=tools,
        base_messages=new_messages,
        temperature=0.3,
        max_retries=1,
        retry_temperature=0.1,
    )

    return reflection_response, completion_tokens, reflection_prompt