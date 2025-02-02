# syntax=docker/dockerfile:1

FROM python:3.12.1-slim

WORKDIR /backend

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000

CMD [ "python" , "app.py","--host=0.0.0.0"]