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
    Plain mode: no signals or diagnostics. Just include previous code and
    instructions to revise. Mirrors the structure used by other modes.
    """
    r = analyze_code(
        original_prompt,
        aux_completion,
        main_completion,
        test_code,
        entry_point,
        num_agent=num_agent,
    )

    # Normalize histories
    if prompt_history_per_agent is None:
        prompt_history_per_agent = [[] for _ in range(int(num_agent))]
    if response_history_per_agent is None:
        response_history_per_agent = [[] for _ in range(int(num_agent))]

    # Single-agent: only main prompt with no aux references
    if int(num_agent) == 1:
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
                f"Revise your {entry_point or 'main'}(...) accordingly. Output ONLY the function code.",
            ]
        )
        return ("", "\n".join(main_lines))

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
            "Revise your aux(...) accordingly. Output ONLY the function code.",
        ]
    )

    main_lines.extend(
        [
            f"Revise your {entry_point or 'main'}(...) accordingly. Output ONLY the function code.",
        ]
    )

    return ("\n".join(aux_lines), "\n".join(main_lines))
