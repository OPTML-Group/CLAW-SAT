FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel

RUN pip install torch==1.6.0 numpy dill tqdm torchtext==0.7.0 tensorboard matplotlib scipy

COPY models/pytorch-seq2seq /model

COPY task/ /app/

ENTRYPOINT [ "/app/entrypoint.sh" ]
