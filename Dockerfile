FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl libglib2.0-0 libgl1 libgomp1 && \
    rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel
RUN python -m pip install --no-cache-dir \
    'numpy<=1.26.4' 'protobuf>=4.21.6,<=4.25.4' \
    psutil ruamel.yaml scipy tqdm opencv-python-headless fast-histogram \
    'onnx>=1.16.1' onnxruntime
RUN python -m pip install --no-cache-dir torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu
RUN python -m pip install --no-cache-dir \
    https://raw.githubusercontent.com/airockchip/rknn-toolkit2/master/rknn-toolkit2/packages/x86_64/rknn_toolkit2-2.3.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

COPY runner.sh /runner.sh
RUN chmod +x /runner.sh
CMD ["/runner.sh"]
