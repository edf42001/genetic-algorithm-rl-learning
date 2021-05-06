FROM python:3-slim as build

# Start the python side of things
WORKDIR /usr/src/app

# Install AWS cli:
RUN apt-get update && apt-get install -y unzip curl && \
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && \
    ./aws/install && \
    rm -rf aws awscliv2.zip && \
    apt-get remove -y unzip curl && \
    apt-get clean

# Install python dependencies. Actual python files are done later
COPY requirements.txt .
RUN pip3 install --no-cache-dir --user --upgrade pip && \
    pip3 install --no-cache-dir --user -r requirements.txt

# Copy remaining python files
COPY . .

# Copy upload script
WORKDIR /usr/src/app
COPY upload_to_s3.sh .

# Run the python agent training code, then upload the results to upload_to_s3
# Store the output to a file so our stdout logs aren't polluted
ENTRYPOINT ./rl_training_enemy_agent.py >> rl_training_enemy_agent_stdout.txt && \
           ./upload_to_s3.sh

# To Run:
#docker run --rm -e AWS_REGION="us-east-1" -e S3_PATH="rl-training-run-results" rl:latest
