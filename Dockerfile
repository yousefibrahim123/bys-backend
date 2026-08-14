FROM python:3.12-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip first to avoid pkg_resources issues
RUN pip install --upgrade pip setuptools wheel

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Create ml_assets directories so the app starts even without trained models.
# The ML endpoints will report available=False but auth/inventory/etc. all work.
RUN mkdir -p ml_assets/models ml_assets/data

EXPOSE ${PORT:-8000}
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
