FROM nvcr.io/nvidia/pytorch:24.03-py3

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    HF_HUB_ENABLE_HL_TRANSFER=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        wget \
        build-essential \
        git-lfs \
        libssl-dev \
        zlib1g-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        libffi-dev \
        libncursesw5-dev \
        xz-utils \
        tk-dev \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://www.python.org/ftp/python/3.12.11/Python-3.12.11.tgz && \
    tar xvf Python-3.12.11.tgz && \
    cd Python-3.12.11 && \
    ./configure --enable-optimizations --prefix=/usr/local && \
    make -j$(nproc) && \
    make altinstall && \
    cd .. && \
    rm -rf Python-3.12.11 Python-3.12.11.tgz

RUN update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python python /usr/local/bin/python3.12 1 && \
    python3 --version

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir flash_attn==2.7.4.post1 --no-build-isolation


CMD ["bash"]
