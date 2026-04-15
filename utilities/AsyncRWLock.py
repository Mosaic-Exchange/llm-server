import contextlib # This library is for managing the parallel generation requests
import asyncio

# This is a custom lock we defined for the following purpose:
# We want to enable parallel generation requests, however there are limitations on that
# Certain generation requests are going to trigger the model reloading to load up that adapter
# So while we can allow parallel generation requests if the requests after the first active one target
# adapters that are already loaded, we cannot allow concurrency for requests that would force
# a reload in the middle of a generation process (sweep the current model out from under a generation request).
# Thus we define a lock to help implement this lock. Additionally we define a maximum number of requests that
# can be active at a given time as to not overwhelm the hardware. This is implemented in a semaphore.
class AsyncRWLock:
    def __init__(self, max_readers: int):
        self._cond = asyncio.Condition()
        self._readers = 0
        self._writer_active = False
        self._writer_waiting = 0
        self._max_readers = max_readers

    @contextlib.asynccontextmanager
    async def reading(self):
        async with self._cond:
            # Check all the conditions for a read process
            # We give writers priority over readers to prevent starvation
            # NOTE: This means that a queued read could come to find that when it gets where it wanted to go
            # the adapter it expected is gone
            while self._writer_active or self._writer_waiting > 0 or self._readers >= self._max_readers:
                await self._cond.wait()
            self._readers += 1 # Increment to count against the MAX_CONCURRENT_GENERATIONS field / value
        try:
            yield
        finally:
            async with self._cond:
                self._readers -= 1 # Decrement to count against the MAX_CONCURRENT_GENERATIONS field / value
                self._cond.notify_all()

    @contextlib.asynccontextmanager
    async def writing(self):
        async with self._cond:
            self._writer_waiting += 1 # Increment to count against the MAX_CONCURRENT_GENERATIONS field / value
            try:
                while self._writer_active or self._readers > 0:
                    await self._cond.wait()
                self._writer_active = True
            finally:
                self._writer_waiting -= 1 # Decrement to count against the MAX_CONCURRENT_GENERATIONS field / value
        try:
            yield
        finally:
            async with self._cond:
                self._writer_active = False
                self._cond.notify_all()
