FROM python:3.9.16-slim-buster
COPY . .
RUN python3 -m pip install --no-cache-dir -r requirements.txt
CMD python FineTuner.py