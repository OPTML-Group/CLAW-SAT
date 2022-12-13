FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel
COPY models/contracode /model
RUN pip install cmake

RUN pip install -e /model

COPY task/ /app/

ENTRYPOINT [ "/app/entrypoint.sh" ]
