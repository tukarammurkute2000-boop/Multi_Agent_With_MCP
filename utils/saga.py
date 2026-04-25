"""
Saga pattern for distributed transactions.
Each step has an execute() and a compensate() (rollback).
If any step fails, all completed steps are rolled back in reverse order.
"""
from dataclasses import dataclass, field
from typing import Callable, Any
import structlog

log = structlog.get_logger()


@dataclass
class SagaStep:
    name: str
    execute: Callable[[], Any]
    compensate: Callable[[Any], None]  # receives the result from execute


class SagaExecutionError(Exception):
    def __init__(self, failed_step: str, original_error: Exception):
        self.failed_step = failed_step
        self.original_error = original_error
        super().__init__(f"Saga failed at step '{failed_step}': {original_error}")


def run_saga(steps: list[SagaStep]) -> list[Any]:
    """
    Execute steps in order. On failure, compensate all completed steps in reverse.
    Returns list of results from each step.
    """
    completed: list[tuple[SagaStep, Any]] = []

    for step in steps:
        try:
            log.info("saga.execute", step=step.name)
            result = step.execute()
            completed.append((step, result))
        except Exception as exc:
            log.error("saga.failed", step=step.name, error=str(exc))
            _compensate_all(completed)
            raise SagaExecutionError(step.name, exc) from exc

    return [r for _, r in completed]


def _compensate_all(completed: list[tuple[SagaStep, Any]]) -> None:
    for step, result in reversed(completed):
        try:
            log.info("saga.compensate", step=step.name)
            step.compensate(result)
        except Exception as exc:
            # Log but don't raise — best-effort compensation
            log.error("saga.compensate_failed", step=step.name, error=str(exc))
