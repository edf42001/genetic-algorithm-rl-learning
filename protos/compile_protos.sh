# set -e

# Generate python classes from protos
python3 -m grpc_tools.protoc -I . --python_out=../python --grpc_python_out=../python hello_world.proto

# Generate java classes from protos
protoc --plugin=protoc-gen-grpc-java --grpc-java_out=. --proto_path=. hello_world.proto