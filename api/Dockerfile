FROM python:3.9-slim

WORKDIR /app

COPY . /app
COPY ../database /app/database
COPY ../model /app/model

COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH="/app:${PYTHONPATH}"

# Commande pour exécuter l'application
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]


