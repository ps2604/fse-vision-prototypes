FROM tensorflow/tensorflow:2.15.0-gpu

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0,1,2,3
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
ENV TF_GPU_MEMORY_GROWTH=true

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3 \
    python3-dev \
    python3-pip \
    python3-distutils \
    build-essential \
    wget \
    curl \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel

COPY requirements.txt /app/
COPY setup.py /app/

RUN pip install -r requirements.txt
RUN pip install -e .

COPY fluxa_fse_native_train.py /app/
COPY fse_native_core.py /app/

RUN mkdir -p /tmp/auralith_fse_native_enhanced

CMD ["python3", "fluxa_fse_native_train.py", "--help"]