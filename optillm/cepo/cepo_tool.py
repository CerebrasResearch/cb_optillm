from typing import Any, Optional, Tuple, List, Dict
from optillm.cepo.cepo import CepoConfig
import copy


# -------------------------------------------------------------------
# Global knobs – you can tweak these centrally
# -------------------------------------------------------------------
CEPO_ENABLE_PLANNING: bool = True          # planning + alignment
CEPO_ENABLE_FINAL_REFLECTION: bool = True  # finish gating
CEPO_ALIGNMENT_INTERVAL: int = 3           # align on step 1 and every N steps
CEPO_FINISH_TOOL_NAME: str = "finish"


# -------------------------------------------------------------------
# Helper utilities
# -------------------------------------------------------------------
def _append_text_to_last_user(messages: List[Dict], extra_text: str) -> None:
    """Append extra_text to the last user message."""
    for m in reversed(messages):
        if m.get("role") == "user":
            content = m.get("content")
            if isinstance(content, str):
                m["content"] = content + extra_text
                return
            if isinstance(content, list):
                if content and isinstance(content[0], dict) and content[0].get("type") == "text":
                    content[0]["text"] = content[0]["text"] + extra_text
                else:
                    content.append({"type": "text", "text": extra_text})
                return
    # If no user message found, silently ignore.


def _extract_between_tags(text: str, start_tag: str, end_tag: str) -> Optional[str]:
    if start_tag not in text or end_tag not in text:
        return None
    try:
        return text.split(start_tag, 1)[1].split(end_tag, 1)[0].strip()
    except Exception:
        return None


def _extract_issue_description(messages: List[Dict]) -> Optional[str]:
    """
    Extract raw issue description assuming SWE-bench-style
    <issue_description>...</issue_description>.
    """
    start_tag = "<issue_description>"
    end_tag = "</issue_description>"
    for m in messages:
        if m.get("role") != "user":
            continue
        content = m.get("content")
        text = None
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            for part in content:
                if part.get("type") == "text":
                    text = part.get("text", "")
                    break
        if not text:
            continue
        inner = _extract_between_tags(text, start_tag, end_tag)
        if inner:
            return inner
    return None


# -------------------------------------------------------------------
# Planning + alignment per step, using cb_log as state
# -------------------------------------------------------------------
def _run_step_with_planning_and_alignment(
    messages: List[Dict],
    client: Any,
    model: str,
    tools: Any,
    cb_log: Dict,
    planning_done: bool,
    spec_plan_text: str,
    alignment_step_idx: int,
) -> Tuple[Any, int, bool, str, int]:
    """
    One CEPO step:

    - If planning not done and CEPO_ENABLE_PLANNING:
        * Ask for spec+plan (no tools), store in spec_plan_text.
    - Increment alignment_step_idx.
    - On certain steps (1 and every CEPO_ALIGNMENT_INTERVAL):
        * Inject an alignment block reminding spec+plan and asking for one tool call.
    - Otherwise:
        * Just call the model with tools.

    Returns:
        response, tokens_used, planning_done, spec_plan_text, alignment_step_idx
    """
    completion_tokens = 0

    # This CEPO call counts as the next "step"
    alignment_step_idx = int(alignment_step_idx) + 1
    cb_log["alignment_step_idx"] = alignment_step_idx

    # If planning is disabled, just passthrough
    if not CEPO_ENABLE_PLANNING:
        kwargs = {"model": model, "messages": messages}
        if tools is not None:
            kwargs["tools"] = tools
        resp = client.chat.completions.create(**kwargs)
        completion_tokens += resp.usage.completion_tokens
        cb_log["planning_done"] = False
        cb_log["alignment_used"] = False
        return resp, completion_tokens, planning_done, spec_plan_text, alignment_step_idx

    # ---------------------------------------------------------
    # First-time planning: get spec+plan (internal only)
    # ---------------------------------------------------------
    if not planning_done:
        spec_plan_prompt = """
Before you run ANY commands or call ANY tools, carefully read the issue description
in this conversation and produce TWO short sections we will reuse:

1) A concise specification / requirements of the bug to fix.
2) A high-level multi-step plan to address it.

Guidelines:
- In the specification, summarize:
  - What behavior is currently broken.
  - What behavior is expected instead.
  - Any inputs / scenarios / edge cases explicitly mentioned.
- In the plan, write 3–7 concrete steps that are realistic in this repo
  (e.g., "Locate parser X", "Add failing repro test", "Adjust parsing of Y", etc.).

Constraints for THIS response:
- DO NOT call any tools.
- DO NOT emit <tool_call> markup.
- Only output natural language describing:
  1) your understanding of the issue, and
  2) a concrete plan to solve it.
"""
        _append_text_to_last_user(messages, spec_plan_prompt)

        plan_resp = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        completion_tokens += plan_resp.usage.completion_tokens
        plan_msg = plan_resp.choices[0].message
        spec_plan_text = plan_msg.content or ""
        planning_done = True

        cb_log["spec_plan_prompt"] = spec_plan_prompt
        cb_log["spec_plan_response"] = spec_plan_text
        cb_log["spec_plan_finish_reason"] = plan_resp.choices[0].finish_reason

        # We do NOT add the planning assistant response into the conversation;
        # it's internal-only guidance.

    # ---------------------------------------------------------
    # Decide whether to inject alignment this step
    # ---------------------------------------------------------
    do_alignment = False
    if spec_plan_text:
        if alignment_step_idx == 1:
            do_alignment = True
        elif CEPO_ALIGNMENT_INTERVAL > 0 and alignment_step_idx % CEPO_ALIGNMENT_INTERVAL == 0:
            do_alignment = True

    cb_log["alignment_used"] = bool(do_alignment)

    if do_alignment and spec_plan_text:
        alignment_suffix = f"""
<ISSUE_SPEC_AND_PLAN>
Here is your earlier specification and high-level plan for this issue:

{spec_plan_text}
</ISSUE_SPEC_AND_PLAN>

<ISSUE_ALIGNMENT_CHECK>
Before deciding your NEXT action, briefly realign with the spec and plan.

1. In 2–4 sentences, say whether your CURRENT understanding of the bug is still
   consistent with the spec. Mention at least one key requirement or scenario
   you have NOT yet fully exercised or verified.
2. State which step of your high-level plan you are currently executing or about
   to execute.
3. Then produce exactly ONE next interaction with the environment
   (a single tool call such as execute_bash, str_replace_editor, think,
    task_tracker, etc.) that moves you toward satisfying an unmet requirement.

Keep the reflection concise; the important output is the next tool call.
</ISSUE_ALIGNMENT_CHECK>
"""
        _append_text_to_last_user(messages, alignment_suffix)
        cb_log["alignment_suffix"] = alignment_suffix

    # ---------------------------------------------------------
    # Actual tool-using step
    # ---------------------------------------------------------
    kwargs = {"model": model, "messages": messages}
    if tools is not None:
        kwargs["tools"] = tools
    resp = client.chat.completions.create(**kwargs)
    completion_tokens += resp.usage.completion_tokens

    cb_log["planning_done"] = planning_done
    cb_log["alignment_finish_reason"] = resp.choices[0].finish_reason

    return resp, completion_tokens, planning_done, spec_plan_text, alignment_step_idx


# -------------------------------------------------------------------
# Final reflection (finish gating), using cb_log as state
# -------------------------------------------------------------------
def _maybe_run_final_reflection(
    base_response: Any,
    messages: List[Dict],
    client: Any,
    model: str,
    tools: Any,
    finish_tool_name: str,
    cb_log: Dict,
    reflection_done: bool,
    spec_plan_text: str,
) -> Tuple[Any, int, bool]:
    """
    If base_response is trying to call the finish tool, and we haven't yet run
    final reflection (reflection_done == False), run one reflection pass that:
      - Reviews issue description + spec/plan text + trajectory so far.
      - Either reaffirms finish or proposes further tool calls.

    On second finish (reflection_done=True), just return base_response.
    """
    completion_tokens = 0

    if reflection_done or not CEPO_ENABLE_FINAL_REFLECTION:
        return base_response, completion_tokens, reflection_done

    msg = base_response.choices[0].message
    tool_calls = getattr(msg, "tool_calls", None) or []

    is_finish = False
    for tc in tool_calls:
        try:
            name = tc.function.name
        except Exception:
            name = None
        if name == finish_tool_name:
            is_finish = True
            break

    if not is_finish:
        return base_response, completion_tokens, reflection_done

    # First finish: intercept and ask for reflection.
    issue_desc = _extract_issue_description(messages) or ""

    cb_log["final_reflection_triggered"] = True
    cb_log["final_reflection_issue_description"] = issue_desc
    cb_log["final_reflection_spec_plan"] = spec_plan_text

    reflection_prompt = f"""
You are now in FINAL REFLECTION mode. The environment run is about to finish.

Your job:
- Double-check whether the current solution truly satisfies the original issue.
- If it does, you may call the finish tool again.
- If it does NOT, you MUST NOT call the finish tool yet; instead, propose the next
  best tool call to continue debugging/fixing.

Information:
<ISSUE_DESCRIPTION>
{issue_desc}
</ISSUE_DESCRIPTION>

<SPEC_AND_PLAN>
{spec_plan_text}
</SPEC_AND_PLAN>

Instructions:
1. Look at the entire conversation above (including your code edits, tests, and
   any previous thoughts) as the trajectory of this attempt.
2. In 2–4 sentences, summarize whether the current behavior and code changes
   satisfy ALL key points in the issue specification and plan.
3. If you are at least reasonably confident (>= 0.8) that everything is correct,
   you may call the finish tool again with an appropriate final message.
4. If you are NOT confident, DO NOT call the finish tool. Instead, choose a
   single best next tool call (e.g., run more tests, inspect a file, adjust code)
   that would most increase your confidence in satisfying the issue.

Important:
- Do NOT start a new plan from scratch; build on what has already been done.
- Do NOT ignore failing tests or missing coverage of key requirements.
"""

    # Build a new message list including the base_response as an assistant turn,
    # then the reflection user prompt.
    try:
        assistant_msg_dict = base_response.choices[0].message.model_dump()
    except Exception:
        assistant_msg_dict = {
            "role": "assistant",
            "content": msg.content,
        }
        if tool_calls:
            assistant_msg_dict["tool_calls"] = [tc.model_dump() for tc in tool_calls]

    messages_for_reflection = list(messages) + [
        assistant_msg_dict,
        {"role": "user", "content": reflection_prompt},
    ]

    kwargs = {"model": model, "messages": messages_for_reflection}
    if tools is not None:
        kwargs["tools"] = tools
    reflection_response = client.chat.completions.create(**kwargs)
    completion_tokens += reflection_response.usage.completion_tokens

    cb_log["final_reflection_finish_reason"] = reflection_response.choices[0].finish_reason
    cb_log["final_reflection_raw_response"] = reflection_response.choices[0].message.content

    reflection_done = True
    return reflection_response, completion_tokens, reflection_done


# -------------------------------------------------------------------
# Main CEPO entry using cb_log in request_config
# -------------------------------------------------------------------
def cepo_tool_v2(
    messages: List[Dict],
    client: Any,
    model: str,
    request_config: dict = None,
) -> Tuple[Any, int]:
    """
    Main CEPO entry.

    - Reads previous cb_log from request_config["cb_log_in"] (if any) and uses it
      as the state blob.
    - Writes updated cb_log back to response.choices[0].cb_log so the caller
      (OpenHands) can persist it and feed it into the next step.
    """
    # ------------------------------
    # Recover previous cb_log (state)
    # ------------------------------
    prev_cb_log: Dict = {}
    tools = None

    if request_config is not None:
        tools = request_config.get("tools")
        prev_cb_log = request_config.get("cb_log", {}) or {}

    # Start new cb_log by copying previous, so state carries forward
    cb_log: Dict = dict(prev_cb_log)

    # Extract persistent state fields from previous cb_log
    planning_done = bool(prev_cb_log.get("planning_done", False))
    spec_plan_text = prev_cb_log.get("spec_plan_text", "") or ""
    alignment_step_idx = int(prev_cb_log.get("alignment_step_idx", 0) or 0)
    reflection_done = bool(prev_cb_log.get("reflection_done", False))

    # Version info
    cb_log["cepo_version"] = 2
    cb_log["cepo_version_description"] = (
        "Planning+alignment every N steps + final reflection; "
        "all state carried via cb_log_in / cb_log."
    )

    # Avoid mutating caller's messages
    messages = copy.deepcopy(messages)

    completion_tokens = 0

    # 1) Main step: planning + alignment
    main_response, tokens_used, planning_done, spec_plan_text, alignment_step_idx = (
        _run_step_with_planning_and_alignment(
            messages=messages,
            client=client,
            model=model,
            tools=tools,
            cb_log=cb_log,
            planning_done=planning_done,
            spec_plan_text=spec_plan_text,
            alignment_step_idx=alignment_step_idx,
        )
    )
    completion_tokens += tokens_used

    # Persist updated state into cb_log
    cb_log["planning_done"] = planning_done
    cb_log["spec_plan_text"] = spec_plan_text
    cb_log["alignment_step_idx"] = alignment_step_idx

    # 2) Optional final reflection on first finish
    main_response, extra_tokens, reflection_done = _maybe_run_final_reflection(
        base_response=main_response,
        messages=messages,
        client=client,
        model=model,
        tools=tools,
        finish_tool_name=CEPO_FINISH_TOOL_NAME,
        cb_log=cb_log,
        reflection_done=reflection_done,
        spec_plan_text=spec_plan_text,
    )
    completion_tokens += extra_tokens

    cb_log["reflection_done"] = reflection_done

    # Attach cb_log to the returned choice for downstream logging/persistence.
    try:
        main_response.choices[0].cb_log = cb_log
    except Exception:
        pass

    return main_response, completion_tokens


# -------------------------------------------------------------------
# Public entrypoint used by optillm / OpenHands
# -------------------------------------------------------------------
def cepo_tool(
    messages: list,
    client: Any,
    model: str,
    cepo_config: CepoConfig,   # currently unused for knobs
    request_config: dict = None,
    request_id: str = None,
):
    """
    Public entry called from optillm.

    - request_config is expected to contain:
        {
          "tools": <OpenAI tools spec>,
          "cb_log": <dict>  # optional: previous cb_log from last step
        }
    - All persistent state is carried via cb_log_in / cb_log.
    """
    return cepo_tool_v2(
        messages=messages,
        client=client,
        model=model,
        request_config=request_config,
    )
