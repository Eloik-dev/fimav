#!/bin/bash
. .venv/bin/activate

OPENCV_VER="master" 
TMPDIR=$(mktemp -d)

# Build and install OpenCV from source.
cd "${TMPDIR}"
git clone --branch ${OPENCV_VER} --depth 1 --recurse-submodules --shallow-submodules https://github.com/opencv/opencv-python.git opencv-python-${OPENCV_VER}
cd opencv-python-${OPENCV_VER}

# Disable headless build, enable contrib and GUI
export ENABLE_CONTRIB=1
export ENABLE_HEADLESS=0

# Pass flags for GStreamer + GTK GUI support
export CMAKE_ARGS="
  -DWITH_GSTREAMER=ON
  -DWITH_GTK=ON
  -DWITH_OPENGL=ON
  -DWITH_V4L=ON
"

pip wheel . --verbose

# Install OpenCV
pip install opencv_python*.whl
