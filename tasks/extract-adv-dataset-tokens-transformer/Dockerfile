FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel

RUN pip install torch==1.7.0 numpy dill tqdm torchtext==0.8.0 tensorboard matplotlib scipy click==7.1.2 allennlp==2.4.0 tensorboard pandas

COPY models/Transformer /model

COPY task/ /app/

ENTRYPOINT [ "/app/entrypoint.sh" ]