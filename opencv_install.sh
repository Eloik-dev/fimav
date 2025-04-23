#!/bin/bash

# Activate your virtual environment
. .venv/bin/activate

sudo apt-get update
sudo apt-get install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly -y
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libfaac-dev libmp3lame-dev libtheora-dev libvorbis-dev -y

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
  -DWITH_FFMPEG=OFF # Explicitly turn OFF FFmpeg
  -DENABLE_PRECOMPILED_HEADERS=OFF # Sometimes helps with build issues
  -DPKG_CONFIG_PATH=/usr/lib/aarch64-linux-gnu/pkgconfig:/usr/share/pkgconfig
"

# Install numpy version compatible with Python 3.11
pip install --upgrade numpy==1.26.4

rm -rf opencv_build/_skbuild

# Install the generated wheel
pip install . --no-build-isolation -v
