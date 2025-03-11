FROM python:3.9-slim

WORKDIR /app

COPY . /app

USER root

RUN pip install --no-cache-dir mlflow

EXPOSE 5000

CMD ["mlflow", "server", "--backend-store-uri", "sqlite:////app/mlflow.db", "--default-artifact-root", "/app/artifacts/mlruns", "--host", "0.0.0.0", "--port", "5000"]