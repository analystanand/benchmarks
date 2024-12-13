import time
import numpy as np
import torch
import tensorflow as tf
import onnxruntime as ort
import matplotlib.pyplot as plt

# Dummy Model Definitions for each framework
class PyTorchModel(torch.nn.Module):
    def __init__(self):
        super(PyTorchModel, self).__init__()
        self.fc = torch.nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

# Create models
pytorch_model = PyTorchModel()
tensorflow_model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(10,))])
tensorflow_model.compile()

# Convert PyTorch model to ONNX
dummy_input = torch.randn(1, 10)
onnx_model_path = "model.onnx"
torch.onnx.export(pytorch_model, dummy_input, onnx_model_path, export_params=True, opset_version=11)
onnx_session = ort.InferenceSession(onnx_model_path)

# Generate dummy data
input_data = np.random.rand(1000, 10).astype(np.float32)

# Function to benchmark a model
def benchmark_model(predict_function, input_data, num_runs=100):
    start_time = time.time()
    for _ in range(num_runs):
        predict_function(input_data)
    end_time = time.time()
    return (end_time - start_time) / num_runs

# Benchmark TensorFlow model
def tensorflow_predict(input_data):
    tensorflow_model(input_data)

tensorflow_time = benchmark_model(lambda x: tensorflow_predict(x), input_data)

# Benchmark PyTorch model
def pytorch_predict(input_data):
    pytorch_model(torch.tensor(input_data))

pytorch_time = benchmark_model(lambda x: pytorch_predict(x), input_data)

# Benchmark ONNX model
def onnx_predict(input_data):
    onnx_session.run(None, {onnx_session.get_inputs()[0].name: input_data})

onnx_time = benchmark_model(lambda x: onnx_predict(x), input_data)

# Plot the results
frameworks = ['TensorFlow', 'PyTorch', 'ONNX']
times = [tensorflow_time, pytorch_time, onnx_time]

plt.bar(frameworks, times)
plt.xlabel('Framework')
plt.ylabel('Average Inference Time (seconds)')
plt.title('Inference Time Comparison')
plt.show()
