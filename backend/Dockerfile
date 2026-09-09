FROM python:3.12-slim
WORKDIR /app
COPY backend/requirements-core.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/app /app/backend/app
COPY backend/models /app/backend/models
COPY frontend /app/frontend
COPY videos /app/videos
ENV ARGUS_RUNTIME_FILE=/app/backend/data/runtime.json
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--app-dir", "backend", "--host", "0.0.0.0", "--port", "8000"]
