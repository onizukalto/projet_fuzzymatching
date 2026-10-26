# On part de l'image officielle d'Airflow
FROM apache/airflow:2.10.5-python3.12

# Outils système : il faut être root pour installer des paquets
USER root
RUN apt-get update \
 && apt-get install -y --no-install-recommends vim \
 && rm -rf /var/lib/apt/lists/*

# On revient à l'utilisateur airflow (Airflow refuse de tourner en root)
USER airflow

# On crée le venv du projet et on y installe nos bibliothèques
COPY requirements.txt /requirements.txt
RUN python -m venv /home/airflow/er-venv \
 && /home/airflow/er-venv/bin/pip install --no-cache-dir -r /requirements.txt