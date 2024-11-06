from typing import Protocol

__all__ = [
    "Generator",
]


class Generator(Protocol):
    """Callable protocol for an LLM generator."""
    def __call__(self, *, prompt: str) -> str:
        ...

