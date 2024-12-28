import time
import os
import numpy as np
import torch
import tensorflow as tf
from tensorflow.keras import Input
import onnxruntime as ort
import matplotlib.pyplot as plt
from PIL import Image
import psutil
import jax
import jax.numpy as jnp
from openvino.runtime import Core
import csv

# Disable GPU and suppress TensorFlow logs
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow logs

# Generate dummy data
input_data = np.random.rand(1000, 200).astype(np.float32)

# Dummy Model Definitions for each framework
class PyTorchModel(torch.nn.Module):
    def __init__(self):
        super(PyTorchModel, self).__init__()
        self.fc1 = torch.nn.Linear(200, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 32)
        self.fc4 = torch.nn.Linear(32, 16)
        self.fc5 = torch.nn.Linear(16, 8)
        self.fc6 = torch.nn.Linear(8, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = self.sigmoid(self.fc6(x))
        return x

# Create PyTorch model
pytorch_model = PyTorchModel()

# TensorFlow Model Definition
tensorflow_model = tf.keras.Sequential([
    Input(shape=(200,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
tensorflow_model.compile()

# JAX Model Definition
def jax_model(x):
    x = jax.nn.relu(jnp.dot(x, jnp.ones((200, 128))))
    x = jax.nn.relu(jnp.dot(x, jnp.ones((128, 64))))
    x = jax.nn.relu(jnp.dot(x, jnp.ones((64, 32))))
    x = jax.nn.relu(jnp.dot(x, jnp.ones((32, 16))))
    x = jax.nn.relu(jnp.dot(x, jnp.ones((16, 8))))
    x = jax.nn.sigmoid(jnp.dot(x, jnp.ones((8, 1))))
    return x

# Convert PyTorch model to ONNX
dummy_input = torch.randn(1, 200)
onnx_model_path = "model.onnx"
torch.onnx.export(
    pytorch_model, 
    dummy_input, 
    onnx_model_path, 
    export_params=True, 
    opset_version=11, 
    input_names=['input'], 
    output_names=['output'], 
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)
onnx_session = ort.InferenceSession(onnx_model_path)

# OpenVINO Model Definition
core = Core()
openvino_model = core.read_model(model="model.onnx")
compiled_model = core.compile_model(openvino_model, device_name="CPU")
# Function to benchmark a model
def benchmark_model(predict_function, input_data, num_runs=1000):
    start_time = time.time()
    process = psutil.Process(os.getpid())
    cpu_usage = []
    memory_usage = []
    for _ in range(num_runs):
        predict_function(input_data)
        cpu_usage.append(process.cpu_percent())
        memory_usage.append(process.memory_info().rss)
    end_time = time.time()
    avg_latency = (end_time - start_time) / num_runs
    avg_cpu = np.mean(cpu_usage)
    avg_memory = np.mean(memory_usage) / (1024 * 1024)  # Convert to MB
    return avg_latency, avg_cpu, avg_memory


# Benchmark PyTorch model
def pytorch_predict(input_data):
    pytorch_model(torch.tensor(input_data))

pytorch_latency, pytorch_cpu, pytorch_memory = benchmark_model(lambda x: pytorch_predict(x), input_data)

# Benchmark TensorFlow model
def tensorflow_predict(input_data):
    tensorflow_model(input_data)

tensorflow_latency, tensorflow_cpu, tensorflow_memory = benchmark_model(lambda x: tensorflow_predict(x), input_data)

# Benchmark JAX model
def jax_predict(input_data):
    jax_model(jnp.array(input_data))

jax_latency, jax_cpu, jax_memory = benchmark_model(lambda x: jax_predict(x), input_data)

# Benchmark ONNX model
def onnx_predict(input_data):
    # Process inputs in batches
    for i in range(input_data.shape[0]):
        single_input = input_data[i:i+1]  # Extract single input
        onnx_session.run(None, {onnx_session.get_inputs()[0].name: single_input})

onnx_latency, onnx_cpu, onnx_memory = benchmark_model(lambda x: onnx_predict(x), input_data)

# Benchmark OpenVINO model
def openvino_predict(input_data):
    # Process inputs in batches
    for i in range(input_data.shape[0]):
        single_input = input_data[i:i+1]  # Extract single input
        compiled_model.infer_new_request({0: single_input})

openvino_latency, openvino_cpu, openvino_memory = benchmark_model(lambda x: openvino_predict(x), input_data)

# Plot the results
frameworks = ['PyTorch', 'TensorFlow', 'JAX', 'ONNX', 'OpenVINO']
latencies = [pytorch_latency, tensorflow_latency, jax_latency, onnx_latency, openvino_latency]
cpu_usages = [pytorch_cpu, tensorflow_cpu, jax_cpu, onnx_cpu, openvino_cpu]
memory_usages = [pytorch_memory, tensorflow_memory, jax_memory, onnx_memory, openvino_memory]

fig, axs = plt.subplots(3, 1, figsize=(10, 15))

axs[0].bar(frameworks, latencies)
axs[0].set_xlabel('Framework')
axs[0].set_ylabel('Average Latency (seconds)')
axs[0].set_title('Latency Comparison')

axs[1].bar(frameworks, cpu_usages)
axs[1].set_xlabel('Framework')
axs[1].set_ylabel('Average CPU Usage (%)')
axs[1].set_title('CPU Usage Comparison')

axs[2].bar(frameworks, memory_usages)
axs[2].set_xlabel('Framework')
axs[2].set_ylabel('Average Memory Usage (MB)')
axs[2].set_title('Memory Usage Comparison')

# Save chart as a JPG image
image_path = "benchmark_comparison.jpg"
plt.savefig(image_path, format='jpg')
plt.show()

# Add architectural details
with Image.open(image_path) as img:
    img.show()

# Calculate relative performance
min_latency = min(latencies)
min_cpu_usage = min(cpu_usages)
min_memory_usage = min(memory_usages)

relative_latencies = [latency / min_latency for latency in latencies]
relative_cpu_usages = [cpu / min_cpu_usage for cpu in cpu_usages]
relative_memory_usages = [memory / min_memory_usage for memory in memory_usages]

# Save results to CSV
csv_file_path = "benchmark_results.csv"
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([
        "Framework", 
        "Latency (seconds)", 
        "CPU Usage (%)", 
        "Memory Usage (MB)", 
        "Relative Latency", 
        "Relative CPU Usage", 
        "Relative Memory Usage"
    ])
    for i in range(len(frameworks)):
        writer.writerow([
            frameworks[i], 
            latencies[i], 
            cpu_usages[i], 
            memory_usages[i], 
            relative_latencies[i], 
            relative_cpu_usages[i], 
            relative_memory_usages[i]
        ])

print(f"Benchmark results saved to {csv_file_path}")