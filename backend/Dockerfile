# FALCON ReID — образ инференса и сервиса.
# Базовый образ уже содержит PyTorch + CUDA runtime, поэтому сборка не тянет
# тяжёлые колёса, а инференс работает полностью офлайн (веса — в составе решения).
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

WORKDIR /app
# libGL для OpenCV (детектор ТС в интерфейсе)
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0     && rm -rf /var/lib/apt/lists/*
ENV PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    HF_HUB_OFFLINE=1 \
    PYTHONPATH=/app

COPY requirements.docker.txt /app/requirements.docker.txt
RUN pip install --no-cache-dir -r requirements.docker.txt

COPY src /app/src
COPY service /app/service
COPY models /app/models

EXPOSE 8000
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]
