# set -e

# Generate python classes from protos
# The 3 level nested protos folders are necessary to make the generated files have the correct import statements
# See: https://github.com/grpc/grpc/issues/9575#issuecomment-293934506
python3 -m grpc_tools.protoc -I protos --python_out=../python --grpc_python_out=../python protos/protos/*.proto

