FROM python:3.7.1

WORKDIR /xception

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt