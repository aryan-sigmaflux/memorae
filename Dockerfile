# Memorae — application image
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# All pinned deps ship manylinux wheels, so no compiler is needed. If a future
# pin ever fails to build on slim, uncomment the build deps below:
# RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
#     && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8003

# Run uvicorn directly (no --reload) for a production container.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003"]
