FROM tensorflow/tensorflow:2.13.0-gpu

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Copy the source code
COPY src/ /app/src/
COPY setup.py .
RUN pip install -e .

ENV PYTHONPATH=/app/src:$PYTHONPATH

ENTRYPOINT ["python", "src/fluxa_fse_native_train.py"]
