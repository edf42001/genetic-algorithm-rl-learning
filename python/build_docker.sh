# Build the python agent code image and tag it
docker build -t rl:python-latest .

# docker tag rl:python-latest edf42001/rl-training:python-latest && docker push edf42001/rl-training:python-latest
# docker run --rm --net=alpine-net -it --name python rl:python-latest