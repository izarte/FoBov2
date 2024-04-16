FROM ubuntu:22.04

WORKDIR /usr/src

RUN apt-get update && apt-get install -y python3 python3-pip mesa-utils libglib2.0-0 libsm6 libxrender1 libxext6

RUN python3 -m pip install --upgrade pip
COPY ./requirements.txt .
RUN python3 -m pip install -r requirements.txt
COPY ./fobo2_env fobo2_env
RUN python3 -m pip install -e fobo2_env
COPY ./train_fobo.py .
CMD [ "python3", "train_fobo.py" ]