FROM python:3.9.16-slim-buster
COPY . .
RUN /usr/local/bin/python -m pip install --upgrade pip
RUN pip install -r requirements.txt
CMD python ./bot/main.py

