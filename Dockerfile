FROM apache/airflow:3.0.2-python3.12

USER root
RUN apt-get update && apt-get install -y tesseract-ocr 

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

