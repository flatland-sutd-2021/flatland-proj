FROM continuumio/anaconda3

WORKDIR /opt/app
COPY . .

RUN /bin/bash -c conda env create -f neurips2020-flatland-starter-kit/environment.yml
RUN /bin/bash -c conda activate flatland-rl
RUN conda install -y pytorch torchvision torchaudio -c pytorch
RUN pip install tensorboard
RUN pip install -U flatland-rl
