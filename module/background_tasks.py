import asyncio
from typing import Set

from .logger_config import logger


class BackgroundTaskManager:
    def __init__(self):
        self._tasks: Set[asyncio.Task] = set()

    def _log_task_completion(self, task: asyncio.Task):
        self._tasks.discard(task)
        try:
            task.result()
        except asyncio.CancelledError:
            logger.debug(f"Task {task.get_name()} was cancelled.")
        except Exception as e:
            logger.error(f"Task {task.get_name()} failed with an exception:", exc_info=e)
        else:
            logger.info(f"Task {task.get_name()} completed successfully. {len(self._tasks)} tasks remaining.")

    def create_task(self, coro):
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._log_task_completion)
        logger.info(f"Task {task.get_name()} started. {len(self._tasks)} tasks running.")
        return task

    async def cancel_all(self):
        if not self._tasks:
            return

        logger.info(f"Cancelling {len(self._tasks)} background tasks.")
        tasks_to_cancel = list(self._tasks)
        for task in tasks_to_cancel:
            if not task.done():
                task.cancel()
        
        await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
        self._tasks.clear()
        logger.info("All background tasks cancelled.")