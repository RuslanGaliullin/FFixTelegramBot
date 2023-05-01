FROM python:3.9.16-slim-buster
COPY . .
RUN pip install -r requirements.txt
CMD python FineTuner.py