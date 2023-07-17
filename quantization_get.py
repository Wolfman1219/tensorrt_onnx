import torch
import torchvision
import onnx
from onnx import quantization

# Load the ONNX model
onnx_model_path = "yolov8n.onnx"
onnx_model = onnx.load(onnx_model_path)

# Check the model
onnx.checker.check_model(onnx_model)

# Quantize the ONNX model
quantized_model = quantization.quantize(onnx_model, quantization_mode=quantization.QuantizationMode.QLinear)

# Save the quantized model
quantized_onnx_model_path = "your_model_quantized.onnx"
onnx.save(quantized_model, quantized_onnx_model_path)
