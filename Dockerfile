# Use an appropriate base image with Python
FROM python:3.9

# Set the working directory inside the Docker container
WORKDIR /app

# Copy the requirements file to the Docker container
COPY requirements.txt .

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Install dependencies from the requirements.txt
RUN pip install --no-cache-dir -r requirements.txt --timeout 120 --retries 5

# Copy the entire project to the Docker container
COPY . .

# Expose the port that Streamlit will run on (default: 8501)
EXPOSE 8501

# Define the command to run the Streamlit app
CMD ["streamlit", "run", "src/Home.py", "--server.port=8501", "--server.address=0.0.0.0"]
