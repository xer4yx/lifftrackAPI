FROM python:3.10-slim as builder
LABEL authors="gameBoyz++"

WORKDIR /app

RUN apt-get update && \
    apt-get install -y \
    git \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


COPY ./requirements.txt ./lifttrackAPI/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r ./requirements.txt

FROM nvidia/cuda:12.0.0-base-ubuntu22.04

WORKDIR /lifttrackAPI

COPY --from=builder /usr/local/lib/python3.10/site-packages/ /usr/local/lib/python3.10/site-packages/

COPY ./model /lifttrackAPI/model
COPY ./config.ini /lifttrackAPI/config.ini
COPY ./main.py /lifttrackAPI/
COPY ./routers /lifttrackAPI/routers
COPY ./lifttrack /lifttrackAPI/lifttrack

RUN echo '#!/bin/bash\n\
# Start Roboflow inference server in the background\n\
roboflow-inference-server-gpu --port 9001 &\n\
\n\
# Wait for Roboflow server to be ready\n\
while ! curl -s http://localhost:9001/health > /dev/null; do\n\
    echo "Waiting for Roboflow server..."\n\
    sleep 2\n\
done\n\
\n\
# Start the FastAPI application\n\
exec uvicorn main:app --host 0.0.0.0 --port 8000\n\
' > /lifttrackAPI/start.sh && chmod +x /lifttrackAPI/start.sh

EXPOSE 8000 9001

CMD ["/lifttrackAPI/start.sh"]