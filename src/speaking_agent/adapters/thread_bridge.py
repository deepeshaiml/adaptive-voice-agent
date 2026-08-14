from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Iterable
import contextlib
from dataclasses import dataclass
from threading import Event
from typing import Any, TypeVar


Item = TypeVar("Item")
_tracked_workers: set[asyncio.Task[Any]] = set()


@dataclass(frozen=True)
class _WorkerFailure:
    error: BaseException


async def iterate_in_thread(
    factory: Callable[[], Iterable[Item]],
    cancellation: Event,
    *,
    stop_timeout_seconds: float = 2.0,
    on_abandoned: Callable[[], None] | None = None,
) -> AsyncIterator[Item]:
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[Item | _WorkerFailure | object] = asyncio.Queue()
    completed = object()

    def emit(item: Item | _WorkerFailure | object) -> None:
        try:
            loop.call_soon_threadsafe(queue.put_nowait, item)
        except RuntimeError:
            pass

    def run() -> None:
        try:
            for item in factory():
                if cancellation.is_set():
                    break
                emit(item)
        except BaseException as error:
            emit(_WorkerFailure(error))
        finally:
            emit(completed)

    worker = asyncio.create_task(asyncio.to_thread(run))
    finished_normally = False
    stop_attempted = False
    try:
        while True:
            item = await queue.get()
            if item is completed:
                break
            if isinstance(item, _WorkerFailure):
                raise item.error
            yield item
        await worker
        finished_normally = True
    except asyncio.CancelledError:
        cancellation.set()
        stop_attempted = True
        stopped = await wait_for_thread_worker(
            worker,
            timeout_seconds=stop_timeout_seconds,
        )
        if not stopped and on_abandoned is not None:
            on_abandoned()
        raise
    finally:
        if not finished_normally and not stop_attempted:
            cancellation.set()
            stopped = await wait_for_thread_worker(
                worker,
                timeout_seconds=stop_timeout_seconds,
            )
            if not stopped and on_abandoned is not None:
                on_abandoned()


async def wait_for_thread_worker(
    worker: asyncio.Task[Any],
    *,
    timeout_seconds: float,
) -> bool:
    try:
        await asyncio.wait_for(asyncio.shield(worker), timeout=timeout_seconds)
        return True
    except TimeoutError:
        _tracked_workers.add(worker)

        def worker_done(completed: asyncio.Task[Any]) -> None:
            _tracked_workers.discard(completed)
            with contextlib.suppress(BaseException):
                completed.exception()

        worker.add_done_callback(worker_done)
        return False
    except BaseException:
        return True
