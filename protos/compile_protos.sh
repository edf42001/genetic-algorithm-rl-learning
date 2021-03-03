# set -e

# Generate python classes from protos
python3 -m grpc_tools.protoc -I . --python_out=../python/protos --grpc_python_out=../python/protos rl_environment_data.proto
