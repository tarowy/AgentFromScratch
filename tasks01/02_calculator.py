import json
import math
import operator
import sys

import openai

from share import env_loader
from share.usage import cost_usd, format_usage

GREEN = "\033[32m"
BLUE = "\033[34m"
RED = "\033[31m"
RESET = "\033[0m"

MAX_STEPS = 5
DEFAULT_QUESTION = "What is 1234 × 5678, minus 999?"

INSTRUCTIONS = (
    "You have a calculator tool. Use it for every arithmetic step, one operation per call, "
    "and wait for each result before using it in the next call. "
    "Never do arithmetic yourself: every number in your answer must come from a tool result."
)

OPERATIONS = {
    "add": operator.add,
    "subtract": operator.sub,
    "multiply": operator.mul,
    "divide": operator.truediv,
}

CALCULATOR_TOOL = {
    "type": "function",
    "name": "calculator",
    "description": "Apply one arithmetic operation to two numbers and return the result: a <operation> b.",
    "parameters": {
        "type": "object",
        "properties": {
            "operation": {"type": "string", "enum": list(OPERATIONS)},
            "a": {"type": "number", "description": "Left operand."},
            "b": {"type": "number", "description": "Right operand."},
        },
        "required": ["operation", "a", "b"],
        "additionalProperties": False,
    },
    "strict": True,
}


def calculator(operation, a, b):
    if operation not in OPERATIONS:
        raise ValueError(f"unknown operation {operation!r}, expected one of {list(OPERATIONS)}")
    for name, value in (("a", a), ("b", b)):
        # bool is a subclass of int, so True would otherwise pass as 1.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a number, got {value!r}")
    result = OPERATIONS[operation](a, b)
    # Float overflow doesn't raise; it quietly returns inf, which the model would then report as an answer.
    if isinstance(result, float) and not math.isfinite(result):
        raise ArithmeticError("result is too large to represent")
    return result


TOOLS = {"calculator": calculator}


def run_tool(call):
    # Never raises: every failure becomes an "Error: ..." string the model can read and react to.
    func = TOOLS.get(call.name)
    if func is None:
        return f"Error: unknown tool {call.name!r}"
    try:
        args = json.loads(call.arguments)
    except json.JSONDecodeError as e:
        return f"Error: arguments are not valid JSON ({e})"
    if not isinstance(args, dict):
        return "Error: arguments must be a JSON object"
    try:
        return str(func(**args))
    except (TypeError, ValueError, ArithmeticError) as e:
        # TypeError: missing or unexpected arguments. ArithmeticError: division by zero, overflow.
        return f"Error: {e}"


def main():
    question = " ".join(sys.argv[1:]) or DEFAULT_QUESTION
    print(f"Question: {question}\n")

    history = [{"role": "user", "content": question}]
    total_cost = 0.0

    for step in range(1, MAX_STEPS + 1):
        try:
            res = env_loader.client.responses.create(
                model=env_loader.model,
                instructions=INSTRUCTIONS,
                tools=[CALCULATOR_TOOL],
                parallel_tool_calls=False,
                input=history,
                store=False,
            )
        except openai.APIError as e:
            print(f"{RED}API error: {e}{RESET}", file=sys.stderr)
            return 1

        if res.status != "completed":
            reason = res.incomplete_details.reason if res.incomplete_details else "unknown"
            print(f"{RED}Response status {res.status!r} (reason: {reason}); stopping.{RESET}", file=sys.stderr)
            return 1

        if res.usage is not None:
            total_cost += cost_usd(res.usage)
            print(f"{BLUE}[step {step}] {format_usage(res.usage)}{RESET}")

        # Append every output item: function_call items must precede their function_call_output.
        history += res.output

        calls = [item for item in res.output if item.type == "function_call"]
        if not calls:
            print(f"\n{GREEN}{res.output_text}{RESET}")
            print(f"Total cost: ${total_cost:.6f}")
            return 0

        # One function_call_output per call_id, all before the next request.
        for call in calls:
            output = run_tool(call)
            print(f"[step {step}] {call.name}({call.arguments}) -> {output}")
            history.append({"type": "function_call_output", "call_id": call.call_id, "output": output})

    print(f"{RED}No final answer after {MAX_STEPS} steps; stopping.{RESET}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
