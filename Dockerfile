#INSTRUCTION TO RUN PYTHON FLASK SERVER API

# Use an official Python runtime as a base image
FROM python:3.11-slim

# Install Java (required for CoreNLP) and wget
RUN apt-get update && apt-get install -y \
    openjdk-21-jre-headless \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy your project files into the container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Stanford CoreNLP
RUN python -c "import stanza; stanza.install_corenlp()"

# Set CoreNLP home environment variable
ENV CORENLP_HOME=/root/stanza_corenlp

# Increase timeout and memory for CoreNLP
ENV CORENLP_TIMEOUT=120000
ENV JAVA_OPTS="-Xmx4g"

#Expose the port
EXPOSE 8080

# Tell Docker what command to run to start the app
# CMD ["python", "app.py"]
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--timeout", "600", "app:app"]


