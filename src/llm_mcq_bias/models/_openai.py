import logging

from openai import OpenAI

__all__ = [
    "gpt_4o_mini",
]

logger = logging.getLogger(__name__)


def gpt_4o_mini(*, prompt: str, options: dict | None = None) -> str:
    # Defaults
    options = options if options is not None else {}

    # openai_gpt_4o_mini
    model = "gpt-4o-mini"

    client = OpenAI()

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            **options,
        )

        logger.debug(f"OpenAI Response: {response}")

        # Parse answer
        answer = response.choices[0].message.content

        return answer
    finally:
        client.close()
