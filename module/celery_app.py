from celery import Celery

from module.config import Config

# Initialize Celery
celery_app = Celery(
    "trading_bot_tasks",
    broker=f"redis://{Config.REDIS_HOST}:{Config.REDIS_PORT}/0",
    backend=f"redis://{Config.REDIS_HOST}:{Config.REDIS_PORT}/1",
    include=["module.tasks"],
)

celery_app.conf.update(
    task_track_started=True,
    result_expires=3600,  # Keep results for 1 hour
    broker_connection_retry_on_startup=True,
)

if __name__ == "__main__":
    celery_app.start()