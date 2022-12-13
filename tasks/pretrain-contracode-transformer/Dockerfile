FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel
COPY models/pytorch-seq2seq /seq2seq
COPY models/contracode /model
RUN pip install cmake==3.22.2
RUN pip install -e /model

COPY task/ /app/

ENTRYPOINT [ "/app/entrypoint.sh" ]
