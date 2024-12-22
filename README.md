# Benchmarks

Compare run time of different frameworks.

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