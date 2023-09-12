# LatenSeer
The repo for SoCC23 Paper: LatenSeer: Causal Modeling of End-to-End Latency Distributions by Harnessing Distributed Tracing.

<div style="text-align: center;">
  <img src="/doc/diagram/latenseer.svg" alt="diagram" width="480"/>
</div>

## What is LatenSeer
LatenSeer is a modeling framework for estimating end-to-end latency distributions in microservice-based web applications.
- An offline tool.
- Harnesses distributed tracing.
- Enables what-if analysis, predicting the potential impacts on end-to-end latency distribution due to various changes in service latencies.

## Repo Structure
The repo includes the [source code](src/) of LatenSeer, [scripts](scripts/) for setting up DSB experiments, and a [script](src/fetch_dsb_traces.py) for collecting DSB Jaeger traces.

### How to use LatenSeer

First, you can create a virtual environment and activate it:
```bash
python -m venv .venv
source .venv/bin/activate
```

Then install LatenSeer:
```bash
pip install .
```

Now you are good to do a test run. We provide a [simple tutorial](examples/simple_tutorial.ipynb) about how to use LatenSeer.


## Citation
```bibtex
@inproceedings{zhang2023-latenseer,
  title={LatenSeer: Causal Modeling of End-to-End Latency Distributions by Harnessing Distributed Tracing},
  author={Zhang, Yazhuo and Isaacs, Rebecca and Yue, Yao and Yang, Juncheng and Zhang, Lei and Vigfusson, Ymir},
  booktitle={ACM Symposium on Cloud Computing (SoCC’23)},
  year={2023}
}