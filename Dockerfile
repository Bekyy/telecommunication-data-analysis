# Use an appropriate base image with Python
# FROM python:3.9

# # Set the working directory inside the Docker container
# WORKDIR /app

# # Copy the requirements file to the Docker container
# COPY requirements.txt .
# RUN pip install --upgrade pip
# # Install dependencies
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy the entire Django project to the Docker container
# COPY . .

# EXPOSE 5432
# Define the command to run the Django app
# CMD ["streamlit", "run", "src/Home.py"]

# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port for MLflow UI
EXPOSE 5000

# Run MLflow tracking server
CMD ["mlflow", "run", "scripts/model.py"]
