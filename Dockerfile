# Use an appropriate base image with Python
FROM python:3.9

# Set the working directory inside the Docker container
WORKDIR /app

# Copy the requirements file to the Docker container
COPY requirements.txt .
RUN pip install --upgrade pip
# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire Django project to the Docker container
COPY . .

EXPOSE 5432
# Define the command to run the Django app
CMD ["streamlit", "run", "src/Home.py"]
