# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory to /app
WORKDIR /app

# Install any needed packages specified in requirements.txt
COPY requirements.txt /app/
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Install necessary system libraries
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && apt-get clean

# Copy the current directory contents into the container at /app
COPY . /app/

# Define environment variable
ENV NAME World
ENV PORT 8501

# Make port 80 available to the world outside this container
EXPOSE $PORT

# Run app.py when the container launches
CMD uvicorn --host 0.0.0.0 --port $PORT app:app
#CMD ["streamlit", "run","dashboard.py"]
