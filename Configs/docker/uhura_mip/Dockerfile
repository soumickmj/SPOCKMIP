# Use an official Anaconda runtime as a parent image
FROM continuumio/anaconda3

# Set the working directory in the container to /app
WORKDIR /uhura_mip

# Add the current directory contents into the container at /app
ADD . /uhura_mip

# Update the environment with any necessary packages specified in requirements.txt
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "uhura_mip", "/bin/bash", "-c"]

# install local dependecies
RUN pip install ./airlab/

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME UHURA_MIP

# Run app.py when the container launches
ENTRYPOINT ["conda", "run", "-n", "uhura_mip", "--no-capture-output", "python", "main_executor.py", "-with_mip", "True", "-model_name", "uhura_mip"]
