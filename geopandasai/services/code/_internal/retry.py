import logging
import random
import time
from typing import Callable, Optional

import litellm

logger = logging.getLogger(__name__)

# Transient errors that are worth retrying. They typically resolve on their own
# (rate limits, momentary overload, flaky network) as opposed to bad requests,
# authentication or context-length errors, which would fail again identically.
RETRYABLE_EXCEPTIONS = (
    litellm.exceptions.RateLimitError,
    litellm.exceptions.ServiceUnavailableError,
    litellm.exceptions.InternalServerError,
    litellm.exceptions.Timeout,
    litellm.exceptions.APIConnectionError,
)

#: Default auto-retry policy. Set ``max_retries`` to 0 (or ``retry_config`` to
#: ``None`` via :func:`update_geopandasai_config`) to disable retries entirely.
DEFAULT_RETRY_CONFIG = {
    "max_retries": 5,
    "initial_delay": 1.0,
    "exponential_base": 2.0,
    "max_delay": 60.0,
    "jitter": True,
}


def completion_with_retry(
    completion_fn: Callable,
    retry_config: Optional[dict],
    **kwargs,
):
    """Call ``completion_fn(**kwargs)`` retrying transient errors with exponential backoff.

    :param completion_fn: The litellm ``completion`` callable (injected so tests
        and custom backends can substitute it).
    :param retry_config: A mapping configuring the backoff (see
        :data:`DEFAULT_RETRY_CONFIG`). When ``None`` or with ``max_retries``
        set to 0, the call is made exactly once with no retry.
    """
    config = {**DEFAULT_RETRY_CONFIG, **(retry_config or {})}
    max_retries = int(config["max_retries"]) if retry_config is not None else 0

    delay = float(config["initial_delay"])
    attempt = 0
    while True:
        try:
            return completion_fn(**kwargs)
        except RETRYABLE_EXCEPTIONS as exc:
            if attempt >= max_retries:
                raise
            wait = min(delay, float(config["max_delay"]))
            if config.get("jitter"):
                wait *= 0.5 + random.random()
            logger.warning(
                "Retrying LLM call after %s (attempt %d/%d), waiting %.1fs",
                type(exc).__name__,
                attempt + 1,
                max_retries,
                wait,
            )
            time.sleep(wait)
            delay *= float(config["exponential_base"])
            attempt += 1
