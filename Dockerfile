FROM continuumio/anaconda3

WORKDIR /opt/app
COPY . .

RUN /bin/bash -c conda env create -f neurips2020-flatland-starter-kit/environment.yml
RUN /bin/bash -c conda activate flatland-rl
RUN conda install -y pytorch torchvision torchaudio -c pytorch
RUN pip install tensorboard boto3
RUN pip install -U flatland-rl

WORKDIR /opt/app/neurips2020-flatland-starter-kit/reinforcement_learning

CMD python multi_agent_training_batch.py
