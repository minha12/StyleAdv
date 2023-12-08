# Dockerfile
FROM nvidia/cuda:11.2.2-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/New_York

# Install Python and other required tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    nano \
    wget \
    curl \
    unzip\
    cmake \
    python3-dev \
    cmake \
    libopenblas-dev liblapack-dev \
    libx11-dev \
    && rm -rf /var/lib/apt/lists/* &&\
    ln -s /usr/bin/python3.8 /usr/bin/python

# Install pip
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

# Install Python packages
COPY requirements.txt /tmp/
RUN pip install --extra-index-url https://download.pytorch.org/whl/cu116 \
    torch==1.13.1 \
    torchvision==0.14.1 \
    -r /tmp/requirements.txt

# Install Ninja
RUN wget https://github.com/ninja-build/ninja/releases/download/v1.8.2/ninja-linux.zip && \
    unzip ninja-linux.zip -d /usr/local/bin/ && \
    update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1 --force && \
    rm ninja-linux.zip

# Clone Git repository
RUN git clone https://github.com/minha12/StyleAdv.git /app

# Set the working directory
WORKDIR /app/

# Run the script to download models
RUN python /app/utils/download_models.py

# # Expose the port on which Gradio runs (usually 7860)
# EXPOSE 7860

# # Set the command to run Gradio app
# ENTRYPOINT ["python", "demo.py"]

