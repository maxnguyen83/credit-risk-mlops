"""The operations console: one page from which bank staff run and read the pipelines.

It starts the training and scoring DAGs through Airflow's REST API, accepts a
CSV of customers to score, and shows the latest call list. It holds no state of
its own -- Airflow is the record of what ran, the shared data directory is the
record of what was produced -- so the container can be restarted at any time.
"""
