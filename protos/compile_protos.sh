# set -e

# Generate python classes from protos
python3 -m grpc_tools.protoc -I . --python_out=../python/protos/generated --grpc_python_out=../python/protos/generated hello_world.proto

