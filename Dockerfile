# Use an official PyTorch runtime as a parent image
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app

# Install any needed packages specified in requirements.txt
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /usr/src/app

# Create a directory for datasets
RUN mkdir /usr/src/data

# Set an environment variable to point to the data directory
ENV DATA_DIR=/usr/src/data
ENV MODEL_PATH=/usr/src/app/default_model
# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# Set the default command to finetune.py, use entrypoint for flexibility
ENTRYPOINT ["python"]
CMD ["finetune.py"]
