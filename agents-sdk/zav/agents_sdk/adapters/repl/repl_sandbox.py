from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar, Dict, Optional

from zav.agents_sdk.domain.agent_dependency import DependencyGroup

HostFunctions = Dict[str, Callable[..., Any]]


class ReplSandbox(ABC):
    """Where `run_python` executes. A sandbox holds one persistent session:
    variables survive between ``run`` calls until ``aclose``. The provider owns
    everything around execution (the tool, the host functions bridging the
    agent's other tools, the prompt, output limits); the sandbox only runs code.
    Monty is the in-process backend; a per-installation remote worker is the
    planned second one."""

    source_name: ClassVar[str]
    # One line for admins choosing sources, e.g. in the agent editor.
    source_description: ClassVar[Optional[str]] = None
    # The tenant gate that hides this source until a tenant enables it.
    source_gate: ClassVar[Optional[str]] = None
    enabled: bool

    @abstractmethod
    async def run(self, code: str, host_functions: HostFunctions) -> str:
        """Execute ``code`` in the session and return its rendered output, or a
        line starting with ``Error:`` or ``SyntaxError:``. Never raises."""
        raise NotImplementedError

    @abstractmethod
    async def to_prompt(self) -> str:
        """What the model must know about this sandbox: filesystem and limits."""
        raise NotImplementedError

    async def aclose(self) -> None:
        return None


class ReplSandboxGroup(DependencyGroup[ReplSandbox]):
    __collects__ = ReplSandbox
