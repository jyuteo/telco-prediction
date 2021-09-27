FROM python:3

WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY /app ./app
COPY /model ./model

EXPOSE 5000