FROM python:3.10
WORKDIR /code
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# HF Spaces expone el puerto 7860 por defecto
CMD ["uvicorn", "api.main_api:app", "--host", "0.0.0.0", "--port", "7860"]
