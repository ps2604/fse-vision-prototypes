FROM tensorflow/tensorflow:2.15.0-gpu

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3 python3-dev python3-pip build-essential libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
COPY setup.py .
RUN pip install -r requirements.txt

# Copy the source code
COPY src/ /app/src/
RUN pip install -e .

CMD ["python3", "src/fluxa_fse_native_train.py", "--help"]
