FROM ubuntu:focal
ADD . /anomaly-detection
VOLUME /home/ubuntu/CTU-13-Dataset:/ctu-13
VOLUME /home/ubuntu/anomaly-detection:/anomaly-detection
WORKDIR ./anomaly-detection
#RUN apk update && apk add python3-dev \
#                        gcc \
#                        libc-dev
RUN apt update -y
RUN apt upgrade -y
RUN apt-get install -y python3.8 \
    && ln -s /usr/bin/python3.8 /usr/bin/python3

RUN apt install python3-pip -y
RUN pip3 install -r ./anomaly-detection/requirements.txt
RUN wandb login
RUN which python3



