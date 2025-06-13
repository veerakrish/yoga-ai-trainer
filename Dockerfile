FROM python:3.9-slim

WORKDIR /code

COPY . .

RUN apt-get update && \
    apt-get install -y \
    build-essential \
    python3-dev \
    python3-pip \
    python3-setuptools \
    gcc \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -r requirements_hf.txt

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
