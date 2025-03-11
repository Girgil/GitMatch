FROM python:3.9-slim

WORKDIR /app

COPY . /app
COPY ../database /app/database
COPY ../model /app/model
COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "/app/model/model_main.py"]
