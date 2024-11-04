import base64
from contextlib import contextmanager
import os
from os import PathLike
import random
import secrets
import string
from typing import Callable

__all__ = [
    "DEFAULT_BUFFER_SIZE",
    "b64decodes",
    "b64encodes",
    "default_arg",
    "pushd",
    "random_string",
]

DEFAULT_BUFFER_SIZE = 2097152
"""Default size to use for IO buffering.

Notes:
    - We use 2MB because that's the Docker default.
"""


def b64encodes(s: str):
    return base64.b64encode(s.encode()).decode()


def b64decodes(s: str):
    return base64.b64decode(s.encode()).decode()


def default_arg[T](
    v: T,
    default: T | None = None,
    default_factory: Callable[[], T] | None = None,
):
    """Populate default parameters."""
    if v is not None:
        return v

    if default is None and default_factory is not None:
        return default_factory()

    return default


def random_string(length: int | None = None):
    """Generate a random string of hexadecimal digits.

    Args:
        length (int): (Optional) The length of the string to generate. Defaults to 8.

    Returns:
        str: a random string of hexadecimal digits
    """
    # Defaults
    length = default_arg(length, 8)

    # Always prefix with a character to ensure value is a string
    return random.choice(string.ascii_lowercase) + secrets.token_hex(int(length / 2))[1:]


@contextmanager
def pushd(new_dir: PathLike):
    """Change directory and restore previous directory on exit."""
    previous_dir = os.getcwd()
    os.chdir(new_dir)
    try:
        yield
    finally:
        os.chdir(previous_dir)
