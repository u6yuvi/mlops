FROM python:3.9.16-slim-buster

RUN pip3 install torch==1.12.1 torchvision==0.13.1 --no-cache-dir --index-url https://download.pytorch.org/whl/cpu
RUN pip3 install black --no-cache-dir
WORKDIR /workspace
COPY train.py /workspace/


CMD ["sh"]
# CMD ["python", "train.py"]