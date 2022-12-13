FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel

RUN pip install torch==1.7.0 numpy dill tqdm torchtext==0.8.0 tensorboard matplotlib scipy tensorboard pandas

COPY models/pytorch-seq2seq /model

COPY task/ /app/

ENTRYPOINT [ "/app/entrypoint.sh" ]