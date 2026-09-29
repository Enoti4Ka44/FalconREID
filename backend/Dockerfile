# FALCON ReID — образ инференса и API-сервиса.
# Базовый образ уже содержит PyTorch + CUDA runtime, поэтому сборка не тянет
# тяжёлые колёса. Во время работы контейнера сеть не нужна: веса — в образе.
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

WORKDIR /app
# libGL/glib для OpenCV (детектор ТС в интерфейсе)
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
ENV PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    HF_HUB_OFFLINE=1 \
    PYTHONPATH=/app

COPY requirements.docker.txt /app/requirements.docker.txt
RUN pip install --no-cache-dir -r requirements.docker.txt

COPY src /app/src
COPY service /app/service
COPY models /app/models

# Веса моделей (~770 МиБ) не помещаются в git (лимит GitHub — 100 МБ на файл).
# Если их нет в ./models, они скачиваются из GitHub-релиза по списку
# models/weights.json, и у каждого файла сверяется sha256. Так разрешено
# постановщиком (Q&A, вопрос 39): сеть допустима при сборке, а запуск
# контейнера идёт офлайн. Если веса уже лежат в ./models, шаг только
# проверяет контрольные суммы.
RUN python -m src.fetch_weights

EXPOSE 8000
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]
