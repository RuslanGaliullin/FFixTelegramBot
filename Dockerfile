FROM python:3.9.16-slim-buster
COPY . .
RUN python3 -m pip install --no-cache-dir -r requirements.txt
RUN python3 -c "from huggingface_hub.hf_api import HfFolder; HfFolder.save_token('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')"
CMD ["python3", "FineTuner.py", "--audio", "data/audio", "--transcript", "data/transcript", "--percent", "0.3", "--evaluation", "1", "--device", "cpu", "--push", "1"]
