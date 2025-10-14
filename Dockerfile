#INSTRUCTION TO RUN PYTHON FLASK SERVER API

# Use an official Python runtime as a base image
FROM python:3.11-slim

WORKDIR /app

# Copy your project files into the container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

#Expose the port
EXPOSE 8080

# Tell Docker what command to run to start the app
# CMD ["python", "app.py"]
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]


