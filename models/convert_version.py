import onnx
from onnx import version_converter
model = onnx.load('./models/face/ultraface.onnx')
model = version_converter.convert_version(model, 12)
onnx.save_model(model, './models/face/ultraface-12.onnx')