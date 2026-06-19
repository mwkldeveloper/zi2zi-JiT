#FROM pytorch/pytorch:2.10.0-cuda13.0-cudnn9-runtime AS base
FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime AS base
RUN apt update && apt -y install libgl1 libglib2.0-0 ssh git python3-venv
#python3-dev apt install build-essential
RUN apt install -y build-essential
WORKDIR /app

COPY ./requirements.txt .
RUN python -m venv --system-site-packages /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN --mount=type=ssh mkdir -p -m 0600 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts
RUN --mount=type=ssh /opt/venv/bin/pip install -r requirements.txt

COPY . .

ENTRYPOINT ["python", "./finetune.py"]
