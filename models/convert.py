import pnnx

# Paths to your ONNX model and desired NCNN output paths
onnx_model_path = "./models/face/ultraface_12.onnx"
ncnn_param_path = "./models/face/ultraface_12.param"
ncnn_bin_path = "./models/face/ultraface_12.bin"

input_shapes = [(1, 3, 240, 320)]  # Batch size 1, 3 color channels, 64x64 resolution
input_types = ['f32']  # The type of data for input (e.g., float32)

# Call the convert function
pnnx.convert(
    ptpath=onnx_model_path,
    ncnnparam=ncnn_param_path,
    ncnnbin=ncnn_bin_path,
    input_shapes=input_shapes,
    input_types=input_types,
    fp16=True  # If you want FP16 precision
)
