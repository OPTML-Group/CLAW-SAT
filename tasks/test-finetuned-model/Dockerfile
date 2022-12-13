FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel
COPY models/contracode /contracode
RUN pip install cmake
RUN pip install -e /contracode

COPY models/pytorch-seq2seq /seq2seq

COPY task/ /app/

ENTRYPOINT [ "/app/entrypoint.sh" ]
