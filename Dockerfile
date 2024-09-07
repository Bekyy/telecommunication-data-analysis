# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the requirements file into the container
COPY requirements.txt ./

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Make port 80 available to the world outside this container (if web app)
EXPOSE 80

# Define environment variable (optional)
ENV NAME World

# Run app.py when the container launches (adjust path if necessary)
CMD ["python", "./src/app.py"]
