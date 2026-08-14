import asyncio
from threading import Event
import unittest

from speaking_agent.adapters.thread_bridge import iterate_in_thread


class ThreadBridgeTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancellation_does_not_wait_forever_for_native_iterator(self) -> None:
        started = Event()
        release = Event()
        abandoned = False

        def factory():
            started.set()
            release.wait()
            yield "done"

        def mark_abandoned() -> None:
            nonlocal abandoned
            abandoned = True

        async def consume() -> None:
            async for _ in iterate_in_thread(
                factory,
                Event(),
                stop_timeout_seconds=0.01,
                on_abandoned=mark_abandoned,
            ):
                pass

        task = asyncio.create_task(consume())
        await asyncio.to_thread(started.wait, 1)
        task.cancel()

        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=0.2)
        self.assertTrue(abandoned)
        release.set()


if __name__ == "__main__":
    unittest.main()