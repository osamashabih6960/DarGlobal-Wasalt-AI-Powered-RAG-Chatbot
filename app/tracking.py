"""Optional MLflow logging. Never breaks the chat if MLflow is unavailable."""

import os

EXPERIMENT = "dar-wasalt-chatbot"


def log_chat(latency_s: float, route: str, n_sources: int) -> None:
    try:
        import mlflow

        if os.getenv("MLFLOW_TRACKING_URI"):
            mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        mlflow.set_experiment(EXPERIMENT + "-live")
        with mlflow.start_run():
            mlflow.log_param("model", os.getenv("MODEL", ""))
            mlflow.log_param("route", route)
            mlflow.log_metric("latency_s", latency_s)
            mlflow.log_metric("n_sources", n_sources)
    except Exception:  # noqa: BLE001
        pass
