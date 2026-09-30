# Level 1 of the reproduction in a container: the package, the notebooks, a CPU torch. (Levels 2 and 3:
# docker/Dockerfile.cuda, ghcr.io/huseyin-karaca/hit-asr:cuda.)
#
#   docker pull ghcr.io/huseyin-karaca/hit-asr:cpu        # or: docker build -f docker/Dockerfile -t hit-asr:cpu .
#   docker run --rm -v "$PWD/out:/out" ghcr.io/huseyin-karaca/hit-asr:cpu            # AMI, the record in ./out
#   docker run --rm -v "$PWD/out:/out" ghcr.io/huseyin-karaca/hit-asr:cpu python -m hitasr.experiment earnings22 --device cpu
FROM python:3.12-slim
LABEL org.opencontainers.image.source="https://github.com/huseyin-karaca/hit-asr" \
      org.opencontainers.image.description="HIT-ASR reproducibility package, level 1 (CPU)"

ENV PIP_NO_CACHE_DIR=1 OMP_NUM_THREADS=1 HITASR_CACHE=/cache
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
RUN python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

WORKDIR /hit-asr
COPY . .
RUN python -m pip install ".[paper]" jupyter

VOLUME ["/out", "/cache"]
WORKDIR /out
CMD ["python", "-m", "hitasr.experiment", "ami_sdm", "--device", "cpu"]
