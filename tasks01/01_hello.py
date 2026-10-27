import sys

import openai

from share import env_loader
from share.usage import cost_usd, format_usage

GREEN = "\033[32m"
BLUE = "\033[34m"
RED = "\033[31m"
RESET = "\033[0m"

TURNS = 2


def read_user_text():
    while True:
        text = input("Input: ").strip()
        if text:
            return text


def main():
    history = []
    total_cost = 0.0

    for _ in range(TURNS):
        try:
            user_text = read_user_text()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        history.append({"role": "user", "content": user_text})

        try:
            res = env_loader.client.responses.create(
                model=env_loader.model,
                input=history,
                store=False,
            )
        except openai.APIError as e:
            print(f"{RED}API error: {e}{RESET}", file=sys.stderr)
            return 1

        print(f"\n{GREEN}{res.model_dump_json(indent=2)}{RESET}\n")
        print("output item types:", [item.type for item in res.output])

        if res.status != "completed":
            # A cut-off reply would leave the history half-finished, so stop rather than continue from it.
            reason = res.incomplete_details.reason if res.incomplete_details else "unknown"
            print(f"{RED}Response status {res.status!r} (reason: {reason}); stopping.{RESET}", file=sys.stderr)
            return 1

        # Resend every output item next turn, not just the text: with store=False the server keeps nothing.
        history += res.output

        print(f"\n{res.output_text}")
        if res.usage is None:
            print(f"{BLUE}[no usage reported]{RESET}")
        else:
            print(f"{BLUE}{format_usage(res.usage)}{RESET}")
            total_cost += cost_usd(res.usage)
        print()

    print(f"Total cost: ${total_cost:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
