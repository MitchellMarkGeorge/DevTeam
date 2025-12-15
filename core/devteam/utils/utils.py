import asyncio
import logging
from copy import deepcopy
from functools import wraps
from typing import Any, Awaitable, Callable, ParamSpec, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def is_dict(obj: Any) -> bool:
    return isinstance(obj, dict)


def merge_dicts(base: dict[str, Any], override: dict[str, Any]):
    # create a deep copy of the base dictionary
    result = deepcopy(base)

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            result[key] = merge_dicts(result[key], value)
        else:
            # Merge non-dictionary values
            result[key] = value
    return result


P = ParamSpec("P")
T = TypeVar("T")


def exponential_backoff_retry(
    delay: float = 2, retries: int = 3
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """
    Decorator that retries an async function with exponential backoff.

    Args:
        delay: Initial delay in seconds between retries (default: 2)
        retries: Maximum number of retry attempts (default: 3)
    """

    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            current_retry = 0
            current_delay = delay
            while current_retry < retries:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    current_retry += 1
                    if current_retry >= retries:
                        raise e
                    print(
                        f"Failed to execute function '{func.__name__}'. Retrying in {current_delay} seconds..."
                    )
                    await asyncio.sleep(current_delay)
                    current_delay *= 2

            # should not reach here
            return await func(*args, **kwargs)

        return wrapper

    return decorator
