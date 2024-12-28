# Benchmarks

This code is used only for information purpose.

Compare run time of different frameworks.

# Environment and Package Versions
The benchmarks were run in the following environment:
```
Experiment Enviroment: [Github Codespace](https://docs.github.com/en/codespaces/overview)
Python Version: 3.12.1
```
The versions of the packages used are:
```
numpy==2.0.2
torch==2.5.1+cpu
tensorflow==2.18.0
onnx==1.17.0
onnxruntime==1.20.1
jax==0.4.38
jaxlib==0.4.38
openvino==2024.6.0
matplotlib==3.9.3
Matplotlib: 3.4.3
Pillow: 8.3.2
psutil: 5.8.0
```
## Setup

1. Create a virtual environment and activate it:

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

2. Install the required dependencies:

    ```sh
    pip install -r requirements.txt
    ```

## Benchmarks for Deep Learning Inference

Benchmarks for deep learning inference measure the performance of different machine learning frameworks when making predictions (inference) using pre-trained models. These benchmarks help in understanding the efficiency and resource utilization of each framework.

### What We Are Testing

We are testing the following aspects of each framework:

1. **Latency**: The average time taken to make a single prediction.
2. **CPU Utilization**: The average percentage of CPU used during the inference process.
3. **Memory Utilization**: The average amount of memory used during the inference process.

The frameworks being tested are:

- TensorFlow
- PyTorch
- ONNX Runtime
- JAX
- OpenVINO

## Running the Benchmark

To run the benchmark and compare the performance of different frameworks, execute the [performance_benchmark.py](http://_vscodecontentref_/1) script:

```sh
python performance_benchmark.py