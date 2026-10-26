from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator

VENV_PYTHON = "/home/airflow/er-venv/bin/python"   # le Python de notre projet, pas celui d'Airflow


def step(module, fonction):
    """Construit la commande qui appelle une fonction de nos fichiers avec le Python du projet."""
    return f'cd /opt/project/src && {VENV_PYTHON} -c "import {module}; {module}.{fonction}()"'


with DAG(
    dag_id="entity_resolution",
    start_date=datetime(2026, 1, 1),
    schedule=None,          # pendant les expériences : uniquement lancé à la main
    catchup=False,
    max_active_runs=1,      # jamais deux exécutions en même temps
    tags=["entity-resolution"],
) as dag:

    t_generate = BashOperator(task_id="generate_data",    bash_command=step("generate_data", "main"))
    t_load     = BashOperator(task_id="load_raw",         bash_command=step("pipeline", "load_raw"))
    t_index    = BashOperator(task_id="index_embeddings", bash_command=step("pipeline", "index_embeddings"))
    t_cands    = BashOperator(task_id="build_candidates", bash_command=step("pipeline", "build_candidates"))
    t_train    = BashOperator(task_id="train_model",      bash_command=step("pipeline", "train_model"))
    t_resolve  = BashOperator(task_id="resolve",          bash_command=step("pipeline", "resolve"))

    t_generate >> t_load >> t_index >> t_cands >> t_train >> t_resolve