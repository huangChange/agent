"""Error taxonomy. Pipeline code branches on these types, never on message strings."""


class GsForgeError(Exception):
    """Base class for all pipeline errors."""


class InputError(GsForgeError):
    """Input data is missing, malformed, or inconsistent. Not retryable."""


class ToolError(GsForgeError):
    """A tool failed on valid input. May be recoverable by the repair agent."""


class DependencyMissingError(ToolError):
    """An optional dependency (GPU stack, pymeshlab, ...) is not installed on this worker."""

    def __init__(self, package: str, extra: str):
        super().__init__(f"optional dependency '{package}' is missing; install with `pip install gsforge[{extra}]`")
        self.package = package
        self.extra = extra


class QAFailedError(GsForgeError):
    """A hard quality gate failed and could not be repaired within budget."""


class BudgetExceededError(GsForgeError):
    """Agent exhausted its turn / token / cost / wall-clock budget."""


class SandboxError(GsForgeError):
    """Agent-generated code failed, timed out, or violated sandbox policy."""
