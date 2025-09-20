from prometheus_client import Counter, Gauge, Histogram

# --- General Metrics ---
APP_INFO = Gauge("app_info", "Application information", ["app_name", "version"])
ERRORS_TOTAL = Counter("app_errors_total", "Total number of errors encountered", ["module", "function"])

# --- Signal Generation Metrics ---
SIGNALS_GENERATED_TOTAL = Counter(
    "signals_generated_total",
    "Total number of trading signals generated",
    ["symbol", "timeframe", "signal_type"],
)
SIGNAL_CONFIDENCE = Histogram(
    "signal_confidence_score",
    "Distribution of confidence scores for generated signals",
    ["signal_type"],
)
ANALYSIS_DURATION_SECONDS = Histogram(
    "analysis_duration_seconds",
    "Duration of symbol analysis in seconds",
    ["timeframe"],
)

# --- Backtesting Metrics ---
BACKTEST_DURATION_SECONDS = Histogram("backtest_duration_seconds", "Duration of backtesting runs")
BACKTEST_RETURN_PERCENT = Gauge("backtest_return_percent", "Return percentage of the last backtest")

# --- Celery Task Metrics ---
CELERY_TASKS_TOTAL = Counter("celery_tasks_total", "Total number of Celery tasks processed", ["task_name", "status"])
CELERY_TASK_DURATION_SECONDS = Histogram("celery_task_duration_seconds", "Duration of Celery tasks", ["task_name"])

# --- API Metrics ---
API_REQUESTS_TOTAL = Counter("api_requests_total", "Total API requests made", ["provider", "endpoint"])
API_REQUEST_LATENCY_SECONDS = Histogram(
    "api_request_latency_seconds", "API request latency", ["provider", "endpoint"]
)


def set_app_info(app_name: str, version: str):
    APP_INFO.labels(app_name=app_name, version=version).set(1)