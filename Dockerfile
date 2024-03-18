FROM python:3.10    

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . .

ENV MAX_JOBS=1
RUN pip install torch transformers pandas tqdm scikit-learn datasets tensorflow numpy --progress-bar off


ENTRYPOINT []
