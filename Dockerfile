# ==============================================================================
# TEKNOFEST 2026 - 5G & Yapay Zeka ile Akilli Yol Guvenligi Yarismasi
# Takim: ANKAAI
# ==============================================================================
# MULTI-STAGE DOCKER BUILD
#
# Stage 1 (builder): Tum bagimliliklari kurar, model donusumu yapar.
#   - PyTorch, Ultralytics ve derleme araclari burada kurulur.
#   - .pt -> .engine (TensorRT) donusumu bu asamada yapilir.
#   - Bu asamanin boyutu onemli degildir.
#
# Stage 2 (runtime): Sadece calisma zamani icin gerekli olanlari kopyalar.
#   - Base Image: nvidia/cuda:12.1.0-base-ubuntu22.04 (ZORUNLU - Sartname §6)
#   - GPU: NVIDIA Tesla T4 | 4 vCPU | 16 GB RAM | 2 GB SHM
#   - Maks. Imaj Boyutu: 8 GB | Maks. Calisma Suresi: 10 dakika
# ==============================================================================

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# STAGE 1: BUILDER
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Sistem bagimliliklari (derleme araclari dahil)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Python bagimliliklarini kur
COPY requirements.txt .
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# [GELECEK] TensorRT Model Donusumu
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Egitilmis modeller hazir oldugunda asagidaki adimlar aktif edilecektir:
#
# COPY models/ /build/models/
# COPY scripts/convert_to_trt.py /build/scripts/
# RUN python3 /build/scripts/convert_to_trt.py \
#     --input /build/models/global_yolov12s.pt \
#     --output /build/models/global_yolov12s.engine \
#     --fp16
# RUN python3 /build/scripts/convert_to_trt.py \
#     --input /build/models/cabin_yolov12n.pt \
#     --output /build/models/cabin_yolov12n.engine \
#     --fp16
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# STAGE 2: RUNTIME
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
FROM nvidia/cuda:12.1.0-base-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Sadece calisma zamani icin gerekli sistem paketleri
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Calisma dizini
WORKDIR /app

# Gerekli klasor yapilari
RUN mkdir -p /app/data/input \
             /app/data/output \
             /app/models \
             /app/src \
             /app/configs

# Builder'dan Python paketlerini kopyala
COPY --from=builder /usr/local/lib/python3.10/dist-packages \
                    /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Model agirliklarini kopyala
# NOT: /app/models/ yolu yarisma spesifikasyonunda belirtilen zorunlu yoldur
COPY models/ /app/models/

# Kaynak kodlarini secici olarak kopyala
# (imaj.tar boyutunu kontrol altinda tutmak icin COPY . . KULLANILMAZ)
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY main.py .
COPY README.md .

# Konteyner ayaga kalktiginda otomatik calisacak komut
CMD ["python3", "main.py"]
