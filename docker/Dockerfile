# Level 1 of the reproduction in a container: the package, the notebooks, a CPU torch.
#
#   docker build -f docker/Dockerfile -t hit-asr .
#   docker run --rm -v "$PWD/out:/out" hit-asr                                              # AMI, the record in ./out
#   docker run --rm -v "$PWD/out:/out" hit-asr python -m hitasr.experiment earnings22 --device cpu
FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 OMP_NUM_THREADS=1 HITASR_CACHE=/cache
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
RUN python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

WORKDIR /hit-asr
COPY . .
RUN python -m pip install ".[paper]" jupyter

VOLUME ["/out", "/cache"]
WORKDIR /out
CMD ["python", "-m", "hitasr.experiment", "ami_sdm", "--device", "cpu"]
