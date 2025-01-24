FROM python:3.9-slim

# Install system dependencies for PyAudio and other packages
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    libasound2-dev \
    portaudio19-dev \
    && apt-get clean

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt /app/

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project directory into the container
COPY . /app/

# Expose the port that Streamlit will run on
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
