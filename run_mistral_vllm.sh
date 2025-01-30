export CUDA_VISIBLE_DEVICES=0

python3 -c "from dotenv import load_dotenv; load_dotenv(); import os; from huggingface_hub import login; login(token=os.getenv('HF_TOKEN'))" && \

python3 -u -m vllm.entrypoints.openai.api_server \
        --model mistralai/Mistral-7B-Instruct-v0.2 \
        --port 3000 \
        --dtype half \
        --tensor-parallel-size 1