import torch
import torch.nn as nn
import onnx
import onnxruntime



PATH = 'model.pt'

# Create an instance of the PyTorch model
model = torch.load(PATH)['model'].float()

# print(model['model'])
# Define an input tensor for the model
input_tensor = torch.randn(1, 3, 640, 640)

# Export the PyTorch model to ONNX format
torch.onnx.export(model, input_tensor, "mymodel.onnx")

# Load the ONNX model using the onnx package
onnx_model = onnx.load("mymodel.onnx")

# Create an instance of the ONNX runtime
ort_session = onnxruntime.InferenceSession("mymodel.onnx")

# Run inference with the ONNX model
ort_inputs = {ort_session.get_inputs()[0].name: input_tensor.numpy()}
ort_outputs = ort_session.run(None, ort_inputs)

# Print the output of the ONNX model
print(ort_outputs)
