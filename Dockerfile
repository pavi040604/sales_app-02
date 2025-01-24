FROM python:3.9-slim

# Install system dependencies required for PyAudio
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    gcc \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy project files into the container
COPY . /app

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
