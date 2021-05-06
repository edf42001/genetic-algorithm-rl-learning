# Build the java environment image and tag it
docker build -t rl:java-latest .

# docker tag rl:java-latest edf42001/rl-training:java-latest && docker push edf42001/rl-training:java-latest
# docker run --rm -e PYTHON_CONTAINER_HOSTNAME="python" --net=alpine-net -it --name java rl:java-latest