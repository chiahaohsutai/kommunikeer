# Kommunikeer - Vision

The Vision service for Kommunikeer is a microservice dedicated to the project’s OCR capabilities — specifically text-detection and text-recognition. After evaluating a variety of open-source, pre-trained vision models, we decided to use PaddlePaddle’s PP-OCRv5 for both detection and recognition. We chose this model because of its superior performance, clear documentation, available SDK support, and ease of integration.

# Local Installation

The text detection and recognition component can be installed in different ways. However, since this project is dockerized and deployed as a standalone service, we will programmatically download the models during the container build (For local development, you can skip this download step — because PaddlePaddle will automatically fetch the models). For simplicity, we will download the models from Hugging Face’s public repository. As a first step, install the Hugging Face CLI (hf), see the [official documentation](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli#command-line-interface-cli) for instructions. After downloading and installing the Hugging Face CLI, please do the following:

```bash
# Navigate to the vision service
cd services/vision

# Download the text-detection and text-recognition models
hf download PaddlePaddle/PP-OCRv5_server_det inference.json inference.pdiparams inference.yml --local-dir ./models/det
hf download PaddlePaddle/PP-OCRv5_server_rec inference.json inference.pdiparams inference.yml --local-dir ./models/rec

# Create Python virtual environment and activate the environment (MacOS)
python -m venv .venv && source .venv/bin/activate

# Install UV and install the project dependencies
pip install uv && uv sync
python -m pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu

# Run the service
hypercorn main:app --workers 4 --log-config json:logging.json --log-level info --access-log - --error-log - -b 0.0.0.0:8000
```

The application will be running on http://localhost:8000 and the OpenAPI UI can be found at http://localhost:8000/api/docs