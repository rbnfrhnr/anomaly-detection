FROM ubuntu:focal
ENV DEBIAN_FRONTEND=noninteractive
ADD . /anomaly-detection
VOLUME /home/ubuntu/CTU-13-Dataset:/ctu-13
VOLUME /home/ubuntu/anomaly-detection:/anomaly-detection
WORKDIR ./anomaly-detection
EXPOSE 8745
RUN apt update -y
RUN apt upgrade -y
RUN apt-get install -y python3.8 \
    && ln -s /usr/bin/python3.8 /usr/bin/python3
RUN apt install tzdata -y
RUN apt install nvidia-cuda-toolkit -y
RUN ./cuda-setup.sh
RUN apt install python3-pip -y
RUN pip3 install -r ./requirements.txt
RUN wandb login
RUN which python3

