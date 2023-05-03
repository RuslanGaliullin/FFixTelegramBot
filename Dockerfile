FROM ubuntu:latest
COPY . .
RUN apt update && apt -y upgrade
RUN apt-get install -y ffmpeg
RUN apt install -y python3-pip
RUN apt install -y g++

RUN pip3 install -r requirements.txt
CMD ["python3", "bot/main.py"]
