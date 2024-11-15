FROM python:3.10-slim as builder
LABEL authors="gameBoyz++"

WORKDIR /app

RUN apt-get update && \
    apt-get install -y git && \
    apt-get clean && \
    rm - rf /var/lib/apt/lists*


COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.10-slim

WORKDIR /lifttrackAPI

COPY --from=builder /usr/local/lib/python3.10/site-packages/ /usr/local/lib/python3.10/site-packages/
COPY ./model /lifttrackAPI/model
COPY ./config.template.ini /lifttrackAPI/config.ini
COPY ./main.py /lifttrackAPI/
COPY ./routers /lifttrackAPI/routers
COPY ./lifttrack /lifttrackAPI/lifttrack

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]