"""
APScheduler-based scheduler for incremental ingestion updates.
Runs ingest_all() on a configurable interval (default: daily at 02:00 UTC).
"""

import logging

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

logger = logging.getLogger(__name__)


class IngestionScheduler:
    def __init__(self, pipeline, vector_store=None):
        self._pipeline = pipeline
        self._vector_store = vector_store
        self._scheduler = BackgroundScheduler(timezone="UTC")

    def _run_ingestion(self) -> None:
        logger.info("Scheduled ingestion started")
        try:
            results = self._pipeline.ingest_all_sync()
            logger.info("Scheduled ingestion complete: %s", results)
        except Exception as exc:
            logger.error("Scheduled ingestion failed: %s", exc, exc_info=True)

    def start(
        self,
        hour: int = 2,
        minute: int = 0,
        day_of_week: str = "*",
    ) -> None:
        self._scheduler.add_job(
            self._run_ingestion,
            trigger=CronTrigger(hour=hour, minute=minute, day_of_week=day_of_week),
            id="ingestion_job",
            replace_existing=True,
        )
        self._scheduler.start()
        logger.info(
            "Ingestion scheduler started (cron: %s:%02d UTC, days=%s)",
            hour, minute, day_of_week,
        )

    def stop(self) -> None:
        self._scheduler.shutdown(wait=False)
        logger.info("Ingestion scheduler stopped")

    def run_now(self) -> None:
        """Trigger an immediate ingestion run (non-blocking)."""
        self._scheduler.add_job(
            self._run_ingestion,
            id="ingestion_now",
            replace_existing=True,
        )
