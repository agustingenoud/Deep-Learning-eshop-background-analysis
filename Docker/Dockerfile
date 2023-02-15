FROM debian:bullseye-slim

RUN apt-get update -y && apt-get install -y curl \
    wget \
    python3 \
    python3-pip \
    python3-dev \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6

RUN apt-get update -y && apt-get install -y
RUN ln -s /usr/bin/python3 /usr/bin/python

COPY requirements.txt /tmp/

RUN pip install --upgrade pip
RUN pip install --requirement /tmp/requirements.txt

RUN apt-get clean
RUN pip cache remove purge

EXPOSE 8889

RUN mkdir -p /workspace

WORKDIR /workspace
CMD jupyter notebook --ip=0.0.0.0 --port=8889 --no-browser --allow-root --NotebookApp.token=