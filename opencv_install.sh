#!/bin/bash

# Activate your virtual environment
. .venv/bin/activate

# Set OpenCV build directory
OPENCV_DIR="./opencv_build"
OPENCV_VER="master"

# Clone only if it doesn't already exist
if [ ! -d "${OPENCV_DIR}" ]; then
    git clone --branch ${OPENCV_VER} --depth 1 --recurse-submodules --shallow-submodules https://github.com/opencv/opencv-python.git "${OPENCV_DIR}"
fi

cd "${OPENCV_DIR}"

# Optional: pull latest changes if needed
# git pull origin ${OPENCV_VER}

# Enable full OpenCV GUI + GStreamer
export ENABLE_CONTRIB=1
export ENABLE_HEADLESS=0
export CMAKE_ARGS="
  -DWITH_GSTREAMER=ON
  -DWITH_GTK=ON
  -DWITH_OPENGL=ON
  -DWITH_V4L=ON
"

# Install numpy version compatible with Python 3.11
pip install --upgrade numpy==1.26.4

# Build the OpenCV wheel
pip wheel . --verbose

# Install the generated wheel
pip install opencv_python*.whl
