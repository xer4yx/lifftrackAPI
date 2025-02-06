# Use an official Ubuntu image as the base
FROM ubuntu:22.04

# Set the working directory in the container
WORKDIR /lifftrackAPI

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip libgl1-mesa-glx libglib2.0-0 libglib2.0-dev \
    ca-certificates git curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy project files into the container
COPY . .

# Ensure the `model` directory exists
RUN mkdir -p /lifftrackAPI/model

# Set PYTHONPATH to ensure modules are found
ENV PYTHONPATH=/lifftrackAPI

# Expose the FastAPI port
EXPOSE 8080

# Entrypoint to start the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]