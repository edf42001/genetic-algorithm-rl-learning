# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: rl_environment_data.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='rl_environment_data.proto',
  package='',
  syntax='proto3',
  serialized_options=b'\n\006protos',
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x19rl_environment_data.proto\"M\n\x0f\x45nvironmentData\x12\r\n\x05state\x18\x01 \x03(\x05\x12\x1a\n\x12last_action_reward\x18\x02 \x01(\x02\x12\x0f\n\x07unit_id\x18\x03 \x01(\x05\" \n\x0e\x41\x63tionResponse\x12\x0e\n\x06\x61\x63tion\x18\x01 \x01(\x05\x32L\n\x12\x45nvironmentService\x12\x36\n\x0fSendEnvironment\x12\x10.EnvironmentData\x1a\x0f.ActionResponse\"\x00\x42\x08\n\x06protosb\x06proto3'
)




_ENVIRONMENTDATA = _descriptor.Descriptor(
  name='EnvironmentData',
  full_name='EnvironmentData',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='state', full_name='EnvironmentData.state', index=0,
      number=1, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='last_action_reward', full_name='EnvironmentData.last_action_reward', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='unit_id', full_name='EnvironmentData.unit_id', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=29,
  serialized_end=106,
)


_ACTIONRESPONSE = _descriptor.Descriptor(
  name='ActionResponse',
  full_name='ActionResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='action', full_name='ActionResponse.action', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=108,
  serialized_end=140,
)

DESCRIPTOR.message_types_by_name['EnvironmentData'] = _ENVIRONMENTDATA
DESCRIPTOR.message_types_by_name['ActionResponse'] = _ACTIONRESPONSE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

EnvironmentData = _reflection.GeneratedProtocolMessageType('EnvironmentData', (_message.Message,), {
  'DESCRIPTOR' : _ENVIRONMENTDATA,
  '__module__' : 'rl_environment_data_pb2'
  # @@protoc_insertion_point(class_scope:EnvironmentData)
  })
_sym_db.RegisterMessage(EnvironmentData)

ActionResponse = _reflection.GeneratedProtocolMessageType('ActionResponse', (_message.Message,), {
  'DESCRIPTOR' : _ACTIONRESPONSE,
  '__module__' : 'rl_environment_data_pb2'
  # @@protoc_insertion_point(class_scope:ActionResponse)
  })
_sym_db.RegisterMessage(ActionResponse)


DESCRIPTOR._options = None

_ENVIRONMENTSERVICE = _descriptor.ServiceDescriptor(
  name='EnvironmentService',
  full_name='EnvironmentService',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=142,
  serialized_end=218,
  methods=[
  _descriptor.MethodDescriptor(
    name='SendEnvironment',
    full_name='EnvironmentService.SendEnvironment',
    index=0,
    containing_service=None,
    input_type=_ENVIRONMENTDATA,
    output_type=_ACTIONRESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_ENVIRONMENTSERVICE)

DESCRIPTOR.services_by_name['EnvironmentService'] = _ENVIRONMENTSERVICE

# @@protoc_insertion_point(module_scope)
