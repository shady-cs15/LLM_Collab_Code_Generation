from typing import Tuple, List, Optional

from .level_feedback import analyze_code


def format_followup_prompts(
    original_prompt: str,
    aux_completion: str,
    main_completion: str,
    test_code: str,
    entry_point: str,
    *,
    num_agent: int = 2,
    prompt_history_per_agent: Optional[List[List[str]]] = None,
    response_history_per_agent: Optional[List[List[str]]] = None,
) -> Tuple[str, str]:
    """
    Produce concise level_passed prompts for each agent using previous code + signals.
    """
    r = analyze_code(
        original_prompt,
        aux_completion,
        main_completion,
        test_code,
        entry_point,
        num_agent=num_agent,
    )

    # Build compact signals
    # Syntax signals
    syntax_sig = "Syntax correct" if r["syntax_ok"] else "Syntax not correct"

    # Test signals
    if r["tests_total"] == 0:
        test_sig = "No tests found"
    elif r["tests_passed"] == r["tests_total"]:
        test_sig = "Passed all tests"
    elif r["tests_passed"] == 0:
        test_sig = "No tests passed"
    else:
        test_sig = f"Passed {r['tests_passed']}/{r['tests_total']} tests"

    if prompt_history_per_agent is None:
        prompt_history_per_agent = [[] for _ in range(int(num_agent))]
    if response_history_per_agent is None:
        response_history_per_agent = [[] for _ in range(int(num_agent))]

    # Single-agent: only main prompt; exclude aux logic/signals
    if int(num_agent) == 1:
        main_impl = (
            "OK"
            if r["main_defined"]
            else "No implementation found in required structure"
        )
        main_lines = []
        # Always include full per-agent history
        if prompt_history_per_agent and prompt_history_per_agent[0]:
            main_lines.extend(["History: previous prompts:"])
            for t, ph in enumerate(prompt_history_per_agent[0], start=1):
                main_lines.append(f"- Turn {t} prompt:\n{ph}")
            main_lines.append("")
        if response_history_per_agent and response_history_per_agent[0]:
            main_lines.extend(["History: your previous responses:"])
            for t, resp in enumerate(response_history_per_agent[0], start=1):
                main_lines.append(f"- Turn {t} response:\n{resp}")
            main_lines.append("")
        main_lines.extend(
            [
                "Signals:",
                f"- Implementation: {main_impl}",
                f"- Syntax: {syntax_sig}",
                f"- Tests: {test_sig}",
                "",
                f"Revise your {entry_point}(...) accordingly. Output ONLY the function code.",
            ]
        )
        return ("", "\n".join(main_lines))

    # Multi-agent (default): original aux + main collaboration
    aux_impl = (
        "OK" if r["aux_defined"] else "No implementation found in required structure"
    )
    main_impl = (
        "OK" if r["main_defined"] else "No implementation found in required structure"
    )

    if not r["main_defined"]:
        aux_use_sig = "Main missing; cannot check aux usage"
    else:
        if not r["aux_called"]:
            aux_use_sig = "Main does not call aux"
        elif r["aux_called_but_not_used"]:
            aux_use_sig = "Aux called but result not used properly"
        else:
            aux_use_sig = "Aux call present and used"

    aux_lines = []
    main_lines = []

    # Always include full per-agent history
    if prompt_history_per_agent and len(prompt_history_per_agent) >= 2:
        aux_ph = prompt_history_per_agent[0]
        main_ph = prompt_history_per_agent[1]
        if aux_ph:
            aux_lines.append("History: previous prompts:")
            for t, ph in enumerate(aux_ph, start=1):
                aux_lines.append(f"- Turn {t} prompt:\n{ph}")
            aux_lines.append("")
        if main_ph:
            main_lines.append("History: previous prompts:")
            for t, ph in enumerate(main_ph, start=1):
                main_lines.append(f"- Turn {t} prompt:\n{ph}")
            main_lines.append("")
    if response_history_per_agent and len(response_history_per_agent) >= 2:
        aux_rh = response_history_per_agent[0]
        main_rh = response_history_per_agent[1]
        if aux_rh:
            aux_lines.append("History: your previous aux(...) responses:")
            for t, resp in enumerate(aux_rh, start=1):
                aux_lines.append(f"- Turn {t} response:\n{resp}")
            aux_lines.append("")
        if main_rh:
            main_lines.append("History: your previous main responses:")
            for t, resp in enumerate(main_rh, start=1):
                main_lines.append(f"- Turn {t} response:\n{resp}")
            main_lines.append("")

    aux_lines.extend(
        [
            "Signals:",
            f"- Implementation: {aux_impl}",
            f"- Syntax: {syntax_sig}",
            f"- Tests: {test_sig}",
            "",
            "Revise your aux(...) accordingly. Output ONLY the function code.",
        ]
    )

    main_lines.extend(
        [
            "Signals:",
            f"- Implementation: {main_impl}",
            f"- Syntax: {syntax_sig}",
            f"- Tests: {test_sig}",
            f"- Aux usage: {aux_use_sig}",
            "",
            f"Revise your {entry_point}(...) accordingly. Output ONLY the function code.",
        ]
    )

    return ("\n".join(aux_lines), "\n".join(main_lines))
