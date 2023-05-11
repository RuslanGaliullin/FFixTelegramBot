FROM python:3.9.16-slim-buster
COPY . .
RUN python3 -m pip install --no-cache-dir -r requirements.txt
CMD ["python3", "FineTuner.py --audio data/audio --transcript data/transcript --percent 0.3 --evaluation 1 --device cpu --push 1"]
