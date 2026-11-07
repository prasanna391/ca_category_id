FROM python:3.11-slim
WORKDIR /app
RUN pip install pandas scipy joblib scikit-learn flask
COPY . .
EXPOSE 5000
ENV PYTHONUNBUFFERED=1
CMD ["python", "src/mlops_pipeline/11_drift_and_retrain.py"]
