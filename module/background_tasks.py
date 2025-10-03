import asyncio
from typing import Set

from .logger_config import logger


class BackgroundTaskManager:
    def __init__(self):
        self._tasks: Set[asyncio.Task] = set()

    def create_task(self, coro):
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        logger.info(f"Task {task.get_name()} started. {len(self._tasks)} tasks running.")
        return task

    async def cancel_all(self):
        if not self._tasks:
            return

        logger.info(f"Cancelling {len(self._tasks)} background tasks.")
        for task in list(self._tasks):
            if not task.done():
                task.cancel()
        
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        logger.info("All background tasks cancelled.")

