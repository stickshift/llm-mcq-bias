import logging

from openai import OpenAI, RateLimitError
import stamina

__all__ = [
    "openai",
]

logger = logging.getLogger(__name__)


@stamina.retry(on=RateLimitError, attempts=None, timeout=300, wait_initial=1, wait_max=60)
def openai(*, model: str, prompt: str, options: dict | None = None) -> str:
    # Defaults
    options = options if options is not None else {}

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

        logger.debug(f"Response: {response}")

        # Parse answer
        answer = response.choices[0].message.content

        return answer
    finally:
        client.close()
