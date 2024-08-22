# Stage 1: Use the GStreamer image to set up GStreamer-related dependencies
FROM restreamio/gstreamer:2022-12-12T14-46-07Z-prod-dbg as gstreamer

# Stage 2: Use the PyTorch base image and copy necessary GStreamer setup from the first stage
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Install additional dependencies
RUN apt-get update && apt-get -y install libgl1 libglib2.0-0 python3 >/dev/null 2>&1
RUN apt-get install -y python3-pip git wget curl ffmpeg libsm6 libxext6 gcc unzip g++ apt-utils >/dev/null 2>&1
RUN apt-get install lsb-release curl gpg -y >/dev/null 2>&1
RUN apt-get install redis -y >/dev/null 2>&1

# Copy GStreamer setup from the previous stage
COPY --from=gstreamer /usr/local /usr/local

# Install GStreamer plugins
RUN apt-get update && apt-get install -y \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-doc \
    gstreamer1.0-tools \
    gstreamer1.0-x \
    gstreamer1.0-alsa \
    gstreamer1.0-pulseaudio

    
RUN pip install psutil

# Ensure PYTHONPATH includes the site-packages directory
ENV PYTHONPATH=/opt/conda/lib/python3.10/site-packages:$PYTHONPATH

# Copy the mvs-drivers.deb file and install it
COPY mvs-drivers.deb mvs-drivers.deb
RUN dpkg -i mvs-drivers.deb

# Set up working directories
WORKDIR ..

RUN mkdir /python

WORKDIR /python

# Install Detectron2 from the GitHub repository
RUN python -m pip install "git+https://x-access-token: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@github.com/Frinks-ai/detectron2"

# Copy and install Python requirements
COPY ./requirements.txt requirements.txt
RUN pip install -r requirements.txt



# # Run unit tests
# CMD ["python", "-m", "unittest", "discover","-s","scripts/tests/", "-v"]


# Run tests using pytest
CMD ["pytest", "scripts/tests/", "-v"]