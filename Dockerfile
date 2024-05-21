FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime AS builder

RUN apt-get update && apt-get install -y git

WORKDIR /app

COPY app.py /app

RUN pip install git+https://github.com/huggingface/parler-tts.git

FROM builder AS runtime

WORKDIR /app

COPY --from=builder /app /app

CMD ["python", "app.py"]