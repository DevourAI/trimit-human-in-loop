import asyncio
import time
from functools import wraps


def rate_limited(interval):
    """
    An asynchronous decorator that prevents a function from being called more than once every
    specified interval. The interval is defined in seconds.
    """

    def decorator(function):
        last_called = [0.0]  # use list to hold mutable last call time

        @wraps(function)
        async def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            if elapsed < interval:
                await asyncio.sleep(interval - elapsed)  # async sleep
            last_called[0] = time.time()
            async for result in function(*args, **kwargs):
                yield result

        return wrapper

    return decorator
