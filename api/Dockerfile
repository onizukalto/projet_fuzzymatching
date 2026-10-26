FROM python:3.12-slim

# Outils système
RUN apt-get update \
 && apt-get install -y --no-install-recommends vim \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/project

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt fastapi uvicorn

CMD ["uvicorn", "main:app", "--app-dir", "/opt/project/api", "--host", "0.0.0.0", "--port", "8000"]