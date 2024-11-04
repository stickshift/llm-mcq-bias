import asyncio as aio
import logging
import shlex
from typing import AsyncIterator, Mapping

from anyio import Path

from ._common import default_arg

__all__ = [
    "ShellCommandError",
    "shell",
    "shell_it",
    "sq",
    "ss",
]

logger = logging.getLogger(__name__)


class ShellCommandError(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return self.message


async def shell_it(
    cmd: str,
    *,
    raise_on_error: bool | None = None,
    capture_stderr: bool | None = None,
    env: Mapping | None = None,
) -> AsyncIterator[str]:
    """Runs shell command and generates output one line at a time."""
    # Defaults
    raise_on_error = default_arg(raise_on_error, True)
    capture_stderr = default_arg(capture_stderr, True)

    # Wrap cmd in bash to ensure we support all bash syntax
    cmd = f"/bin/bash -c {shlex.quote(cmd)}"

    # Launch subprocess shell
    proc = await aio.create_subprocess_shell(
        cmd,
        stdin=aio.subprocess.PIPE,
        stdout=aio.subprocess.PIPE,
        stderr=(aio.subprocess.STDOUT if capture_stderr else None),
        env=env,
    )

    # Explicitly close stdin to tell process there is nothing to read
    proc.stdin.close()

    # Stream output one line at a time
    while data := await proc.stdout.readline():
        yield data.decode().rstrip()

    # Wait for process to finish
    await proc.wait()

    if proc.returncode != 0 and raise_on_error:
        raise ShellCommandError(f"Command '{cmd}' failed.")


async def shell(
    cmd: str,
    *,
    raise_on_error: bool | None = None,
    capture_stderr: bool | None = None,
    env: Mapping | None = None,
) -> str | dict:
    """Runs shell command and returns output."""
    it = shell_it(
        cmd, raise_on_error=raise_on_error, capture_stderr=capture_stderr, env=env
    )

    output_lines = []
    async for line in it:
        logger.debug(line)
        output_lines.append(line)

    output = "\n".join(output_lines)

    return output


def ss(v: str) -> list[str]:
    """Splits shell into tokens."""
    return shlex.split(v)


def sq(v: str | Path) -> str:
    """Wraps v in shell quotes."""
    return shlex.quote(str(v))
