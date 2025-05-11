import pnnx

# Chemins vers le modèle ONNX et les fichiers NCNN de sortie
onnx_model_path = "./models/face/ultraface_12.onnx"
ncnn_param_path = "./models/face/ultraface_12.param"
ncnn_bin_path = "./models/face/ultraface_12.bin"

input_shapes = [(1, 3, 240, 320)] 
input_types = ['f32']

# Convertir le modèle ONNX
pnnx.convert(
    ptpath=onnx_model_path,
    ncnnparam=ncnn_param_path,
    ncnnbin=ncnn_bin_path,
    input_shapes=input_shapes,
    input_types=input_types,
    fp16=True
)
