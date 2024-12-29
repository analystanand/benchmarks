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
from plotly.subplots import make_subplots
import plotly.graph_objects as go

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

# prepare data for chart
frameworks = ['PyTorch', 'TensorFlow', 'JAX', 'ONNX', 'OpenVINO']
latencies = [pytorch_latency, tensorflow_latency, jax_latency, onnx_latency, openvino_latency]
cpu_usages = [pytorch_cpu, tensorflow_cpu, jax_cpu, onnx_cpu, openvino_cpu]
memory_usages = [pytorch_memory, tensorflow_memory, jax_memory, onnx_memory, openvino_memory]

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
        "Latency (milliseconds)", 
        "CPU Usage (%)", 
        "Memory Usage (MB)", 
        "Relative Latency", 
        "Relative CPU Usage", 
        "Relative Memory Usage"
    ])
    for i in range(len(frameworks)):
        writer.writerow([
            frameworks[i], 
            latencies[i] * 1000,  # Convert latency to milliseconds
            cpu_usages[i], 
            memory_usages[i], 
            relative_latencies[i], 
            relative_cpu_usages[i], 
            relative_memory_usages[i]
        ])

print(f"Benchmark results saved to {csv_file_path}")

# Visualize the results
# Create subplots
fig = make_subplots(rows=3, cols=1)
# Add latency bar chart
fig.add_trace(go.Bar(x=frameworks, y=[latency * 1000 for latency in latencies], name='Latency (milliseconds)', showlegend=True, legendgroup='latency'), row=1, col=1)
fig.add_trace(go.Scatter(x=frameworks, y=relative_latencies, mode='lines+markers', name='Relative Latency', showlegend=True, legendgroup='latency'), row=1, col=1)
fig.update_xaxes(title_text="Frameworks", row=1, col=1)
fig.update_yaxes(title_text="Latency (ms)", row=1, col=1)
fig.add_annotation(text="<b>Latency Comparison</b>", xref="x domain", yref="y domain", x=0.5, y=1.1, showarrow=False, row=1, col=1)

# Add CPU usage bar chart
fig.add_trace(go.Bar(x=frameworks, y=cpu_usages, name='CPU Usage (%)', showlegend=True, legendgroup='cpu'), row=2, col=1)
fig.add_trace(go.Scatter(x=frameworks, y=relative_cpu_usages, mode='lines+markers', name='Relative CPU Usage', showlegend=True, legendgroup='cpu'), row=2, col=1)
fig.update_xaxes(title_text="Frameworks", row=2, col=1)
fig.update_yaxes(title_text="CPU Usage (%)", row=2, col=1)
fig.add_annotation(text="<b>CPU Usage Comparison</b>", xref="x domain", yref="y domain", x=0.5, y=1.1, showarrow=False, row=2, col=1)

# Add memory usage bar chart
fig.add_trace(go.Bar(x=frameworks, y=memory_usages, name='Memory Usage (MB)', showlegend=True, legendgroup='memory'), row=3, col=1)
fig.add_trace(go.Scatter(x=frameworks, y=relative_memory_usages, mode='lines+markers', name='Relative Memory Usage', showlegend=True, legendgroup='memory'), row=3, col=1)
fig.update_xaxes(title_text="Frameworks", row=3, col=1)
fig.update_yaxes(title_text="Memory Usage (MB)", row=3, col=1)
fig.add_annotation(text="<b>Memory Usage Comparison</b>", xref="x domain", yref="y domain", x=0.5, y=1.1, showarrow=False, row=3, col=1)

# Update layout
fig.update_layout(
    height=900, 
    width=1000,  # Increased width to accommodate legends
    title={
        'text': "Comparison of Deep Learning Inference Frameworks<br><span style='font-size:14px; color:gray;'>Latency, CPU Usage and Memory Usage</span>",
        'y':0.96,  # Positioned close to the top
        'x':0.4,
        'xanchor': 'center',
        'yanchor': 'top'
    },
    margin=dict(t=120, b=50, l=50, r=150),  # Increased right margin to accommodate legends
    legend=dict(
        x=1.1,  # Position legend to the right of the subplots
        y=0.5,  # Center the legend vertically
        yanchor='middle'
    ),
    legend_tracegroupgap=180
)

# Save chart as an HTML file
html_path = "benchmark_comparison.html"
fig.write_html(html_path)
print(f"Chart saved to {html_path}")