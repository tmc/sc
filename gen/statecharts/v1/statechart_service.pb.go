// Statechart Service.
//
// This file defines a service to execute statechart semantics.
//

// Code generated by protoc-gen-go. DO NOT EDIT.
// versions:
// 	protoc-gen-go v1.28.0
// 	protoc        (unknown)
// source: statecharts/v1/statechart_service.proto

package statechartsv1

import (
	status "google.golang.org/genproto/googleapis/rpc/status"
	protoreflect "google.golang.org/protobuf/reflect/protoreflect"
	protoimpl "google.golang.org/protobuf/runtime/protoimpl"
	structpb "google.golang.org/protobuf/types/known/structpb"
	reflect "reflect"
	sync "sync"
)

const (
	// Verify that this generated code is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(20 - protoimpl.MinVersion)
	// Verify that runtime/protoimpl is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(protoimpl.MaxVersion - 20)
)

// A registry of Statecharts.
type StatechartRegistry struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	// The registry of Statecharts.
	Statecharts map[string]*Statechart `protobuf:"bytes,1,rep,name=statecharts,proto3" json:"statecharts,omitempty" protobuf_key:"bytes,1,opt,name=key,proto3" protobuf_val:"bytes,2,opt,name=value,proto3"`
}

func (x *StatechartRegistry) Reset() {
	*x = StatechartRegistry{}
	if protoimpl.UnsafeEnabled {
		mi := &file_statecharts_v1_statechart_service_proto_msgTypes[0]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *StatechartRegistry) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*StatechartRegistry) ProtoMessage() {}

func (x *StatechartRegistry) ProtoReflect() protoreflect.Message {
	mi := &file_statecharts_v1_statechart_service_proto_msgTypes[0]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use StatechartRegistry.ProtoReflect.Descriptor instead.
func (*StatechartRegistry) Descriptor() ([]byte, []int) {
	return file_statecharts_v1_statechart_service_proto_rawDescGZIP(), []int{0}
}

func (x *StatechartRegistry) GetStatecharts() map[string]*Statechart {
	if x != nil {
		return x.Statecharts
	}
	return nil
}

// A request to create a new machine.
type CreateMachineRequest struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	// The ID of the statechart to create an instance from.
	StatechartId string `protobuf:"bytes,1,opt,name=statechart_id,json=statechartId,proto3" json:"statechart_id,omitempty"`
	// The initial context of the machine.
	Context *structpb.Struct `protobuf:"bytes,2,opt,name=context,proto3" json:"context,omitempty"`
}

func (x *CreateMachineRequest) Reset() {
	*x = CreateMachineRequest{}
	if protoimpl.UnsafeEnabled {
		mi := &file_statecharts_v1_statechart_service_proto_msgTypes[1]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *CreateMachineRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*CreateMachineRequest) ProtoMessage() {}

func (x *CreateMachineRequest) ProtoReflect() protoreflect.Message {
	mi := &file_statecharts_v1_statechart_service_proto_msgTypes[1]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use CreateMachineRequest.ProtoReflect.Descriptor instead.
func (*CreateMachineRequest) Descriptor() ([]byte, []int) {
	return file_statecharts_v1_statechart_service_proto_rawDescGZIP(), []int{1}
}

func (x *CreateMachineRequest) GetStatechartId() string {
	if x != nil {
		return x.StatechartId
	}
	return ""
}

func (x *CreateMachineRequest) GetContext() *structpb.Struct {
	if x != nil {
		return x.Context
	}
	return nil
}

// A response to a create machine request.
type CreateMachineResponse struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	// The created machine.
	Machine *Machine `protobuf:"bytes,1,opt,name=machine,proto3" json:"machine,omitempty"`
}

func (x *CreateMachineResponse) Reset() {
	*x = CreateMachineResponse{}
	if protoimpl.UnsafeEnabled {
		mi := &file_statecharts_v1_statechart_service_proto_msgTypes[2]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *CreateMachineResponse) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*CreateMachineResponse) ProtoMessage() {}

func (x *CreateMachineResponse) ProtoReflect() protoreflect.Message {
	mi := &file_statecharts_v1_statechart_service_proto_msgTypes[2]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use CreateMachineResponse.ProtoReflect.Descriptor instead.
func (*CreateMachineResponse) Descriptor() ([]byte, []int) {
	return file_statecharts_v1_statechart_service_proto_rawDescGZIP(), []int{2}
}

func (x *CreateMachineResponse) GetMachine() *Machine {
	if x != nil {
		return x.Machine
	}
	return nil
}

// StepRequest is the request message for the Step method.
type StepRequest struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	// The id of the statechart to step.
	StatechartId string `protobuf:"bytes,1,opt,name=statechart_id,json=statechartId,proto3" json:"statechart_id,omitempty"`
	// The event to step the statechart with.
	Event string `protobuf:"bytes,2,opt,name=event,proto3" json:"event,omitempty"` // The context attached to the Event.
}

func (x *StepRequest) Reset() {
	*x = StepRequest{}
	if protoimpl.UnsafeEnabled {
		mi := &file_statecharts_v1_statechart_service_proto_msgTypes[3]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *StepRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*StepRequest) ProtoMessage() {}

func (x *StepRequest) ProtoReflect() protoreflect.Message {
	mi := &file_statecharts_v1_statechart_service_proto_msgTypes[3]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use StepRequest.ProtoReflect.Descriptor instead.
func (*StepRequest) Descriptor() ([]byte, []int) {
	return file_statecharts_v1_statechart_service_proto_rawDescGZIP(), []int{3}
}

func (x *StepRequest) GetStatechartId() string {
	if x != nil {
		return x.StatechartId
	}
	return ""
}

func (x *StepRequest) GetEvent() string {
	if x != nil {
		return x.Event
	}
	return ""
}

// StepResponse is the response message for the Step method.
type StepResponse struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	// The statechart's current state.
	Machine *Machine `protobuf:"bytes,1,opt,name=machine,proto3" json:"machine,omitempty"`
	// The result of the step operation.
	Result *status.Status `protobuf:"bytes,2,opt,name=result,proto3" json:"result,omitempty"`
}

func (x *StepResponse) Reset() {
	*x = StepResponse{}
	if protoimpl.UnsafeEnabled {
		mi := &file_statecharts_v1_statechart_service_proto_msgTypes[4]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *StepResponse) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*StepResponse) ProtoMessage() {}

func (x *StepResponse) ProtoReflect() protoreflect.Message {
	mi := &file_statecharts_v1_statechart_service_proto_msgTypes[4]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use StepResponse.ProtoReflect.Descriptor instead.
func (*StepResponse) Descriptor() ([]byte, []int) {
	return file_statecharts_v1_statechart_service_proto_rawDescGZIP(), []int{4}
}

func (x *StepResponse) GetMachine() *Machine {
	if x != nil {
		return x.Machine
	}
	return nil
}

func (x *StepResponse) GetResult() *status.Status {
	if x != nil {
		return x.Result
	}
	return nil
}

var File_statecharts_v1_statechart_service_proto protoreflect.FileDescriptor

var file_statecharts_v1_statechart_service_proto_rawDesc = []byte{
	0x0a, 0x27, 0x73, 0x74, 0x61, 0x74, 0x65, 0x63, 0x68, 0x61, 0x72, 0x74, 0x73, 0x2f, 0x76, 0x31,
	0x2f, 0x73, 0x74, 0x61, 0x74, 0x65, 0x63, 0x68, 0x61, 0x72, 0x74, 0x5f, 0x73, 0x65, 0x72, 0x76,
	0x69, 0x63, 0x65, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x12, 0x0e, 0x73, 0x74, 0x61, 0x74, 0x65,
	0x63, 0x68, 0x61, 0x72, 0x74, 0x73, 0x2e, 0x76, 0x31, 0x1a, 0x1c, 0x67, 0x6f, 0x6f, 0x67, 0x6c,
	0x65, 0x2f, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x62, 0x75, 0x66, 0x2f, 0x73, 0x74, 0x72, 0x75, 0x63,
	0x74, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x1a, 0x17, 0x67, 0x6f, 0x6f, 0x67, 0x6c, 0x65, 0x2f,
	0x72, 0x70, 0x63, 0x2f, 0x73, 0x74, 0x61, 0x74, 0x75, 0x73, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f,
	0x1a, 0x20, 0x73, 0x74, 0x61, 0x74, 0x65, 0x63, 0x68, 0x61, 0x72, 0x74, 0x73, 0x2f, 0x76, 0x31,
	0x2f, 0x73, 0x74, 0x61, 0x74, 0x65, 0x63, 0x68, 0x61, 0x72, 0x74, 0x73, 0x2e, 0x70, 0x72, 0x6f,
	0x74, 0x6f, 0x22, 0xc7, 0x01, 0x0a, 0x12, 0x53, 0x74, 0x61, 0x74, 0x65, 0x63, 0x68, 0x61, 0x72,
	0x74, 0x52, 0x65, 0x67, 0x69, 0x73, 0x74, 0x72, 0x79, 0x12, 0x55, 0x0a, 0x0b, 0x73, 0x74, 0x61,
	0x74, 0x65, 0x63, 0x68, 0x61, 0x72, 0x74, 0x73, 0x18, 0x01, 0x20, 0x03, 0x28, 0x0b, 0x32, 0x33,
	0x2e, 0x73, 0x74, 0x61, 0x74, 0x65, 0x63, 0x68, 0x61, 0x72, 0x74, 0x73, 0x2e, 0x76, 0x31, 0x2e,
	0x53, 0x74, 0x61, 0x74, 0x65, 0x63, 0x68, 0x61, 0x72, 0x74, 0x52, 0x65, 0x67, 0x69, 0x73, 0x74,
	0x72, 0x79, 0x2e, 0x53, 0x74, 0x61, 0x74, 0x65, 0x63, 0x68, 0x61, 0x72, 0x74, 0x73, 0x45, 0x6e,
	0x74, 0x72, 0x79, 0x52, 0x0b, 0x73, 0x74, 0x61, 0x74, 0x65, 0x63, 0x68, 0x61, 0x72, 0x74, 0x73,
	0x1a, 0x5a, 0x0a, 0x10, 0x53, 0x74, 0x61, 0x74, 0x65, 0x63, 0x68, 0x61, 0x72, 0x74, 0x73, 0x45,
	0x6e, 0x74, 0x72, 0x79, 0x12, 0x10, 0x0a, 0x03, 0x6b, 0x65, 0x79, 0x18, 0x01, 0x20, 0x01, 0x28,
	0x09, 0x52, 0x03, 0x6b, 0x65, 0x79, 0x12, 0x30, 0x0a, 0x05, 0x76, 0x61, 0x6c, 0x75, 0x65, 0x18,
	0x02, 0x20, 0x01, 0x28, 0x0b, 0x32, 0x1a, 0x2e, 0x73, 0x74, 0x61, 0x74, 0x65, 0x63, 0x68, 0x61,
	0x72, 0x74, 0x73, 0x2e, 0x76, 0x31, 0x2e, 0x53, 0x74, 0x61, 0x74, 0x65, 0x63, 0x68, 0x61, 0x72,
	0x74, 0x52, 0x05, 0x76, 0x61, 0x6c, 0x75, 0x65, 0x3a, 0x02, 0x38, 0x01, 0x22, 0x6e, 0x0a, 0x14,
	0x43, 0x72, 0x65, 0x61, 0x74, 0x65, 0x4d, 0x61, 0x63, 0x68, 0x69, 0x6e, 0x65, 0x52, 0x65, 0x71,
	0x75, 0x65, 0x73, 0x74, 0x12, 0x23, 0x0a, 0x0d, 0x73, 0x74, 0x61, 0x74, 0x65, 0x63, 0x68, 0x61,
	0x72, 0x74, 0x5f, 0x69, 0x64, 0x18, 0x01, 0x20, 0x01, 0x28, 0x09, 0x52, 0x0c, 0x73, 0x74, 0x61,
	0x74, 0x65, 0x63, 0x68, 0x61, 0x72, 0x74, 0x49, 0x64, 0x12, 0x31, 0x0a, 0x07, 0x63, 0x6f, 0x6e,
	0x74, 0x65, 0x78, 0x74, 0x18, 0x02, 0x20, 0x01, 0x28, 0x0b, 0x32, 0x17, 0x2e, 0x67, 0x6f, 0x6f,
	0x67, 0x6c, 0x65, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x62, 0x75, 0x66, 0x2e, 0x53, 0x74, 0x72,
	0x75, 0x63, 0x74, 0x52, 0x07, 0x63, 0x6f, 0x6e, 0x74, 0x65, 0x78, 0x74, 0x22, 0x4a, 0x0a, 0x15,
	0x43, 0x72, 0x65, 0x61, 0x74, 0x65, 0x4d, 0x61, 0x63, 0x68, 0x69, 0x6e, 0x65, 0x52, 0x65, 0x73,
	0x70, 0x6f, 0x6e, 0x73, 0x65, 0x12, 0x31, 0x0a, 0x07, 0x6d, 0x61, 0x63, 0x68, 0x69, 0x6e, 0x65,
	0x18, 0x01, 0x20, 0x01, 0x28, 0x0b, 0x32, 0x17, 0x2e, 0x73, 0x74, 0x61, 0x74, 0x65, 0x63, 0x68,
	0x61, 0x72, 0x74, 0x73, 0x2e, 0x76, 0x31, 0x2e, 0x4d, 0x61, 0x63, 0x68, 0x69, 0x6e, 0x65, 0x52,
	0x07, 0x6d, 0x61, 0x63, 0x68, 0x69, 0x6e, 0x65, 0x22, 0x48, 0x0a, 0x0b, 0x53, 0x74, 0x65, 0x70,
	0x52, 0x65, 0x71, 0x75, 0x65, 0x73, 0x74, 0x12, 0x23, 0x0a, 0x0d, 0x73, 0x74, 0x61, 0x74, 0x65,
	0x63, 0x68, 0x61, 0x72, 0x74, 0x5f, 0x69, 0x64, 0x18, 0x01, 0x20, 0x01, 0x28, 0x09, 0x52, 0x0c,
	0x73, 0x74, 0x61, 0x74, 0x65, 0x63, 0x68, 0x61, 0x72, 0x74, 0x49, 0x64, 0x12, 0x14, 0x0a, 0x05,
	0x65, 0x76, 0x65, 0x6e, 0x74, 0x18, 0x02, 0x20, 0x01, 0x28, 0x09, 0x52, 0x05, 0x65, 0x76, 0x65,
	0x6e, 0x74, 0x22, 0x6d, 0x0a, 0x0c, 0x53, 0x74, 0x65, 0x70, 0x52, 0x65, 0x73, 0x70, 0x6f, 0x6e,
	0x73, 0x65, 0x12, 0x31, 0x0a, 0x07, 0x6d, 0x61, 0x63, 0x68, 0x69, 0x6e, 0x65, 0x18, 0x01, 0x20,
	0x01, 0x28, 0x0b, 0x32, 0x17, 0x2e, 0x73, 0x74, 0x61, 0x74, 0x65, 0x63, 0x68, 0x61, 0x72, 0x74,
	0x73, 0x2e, 0x76, 0x31, 0x2e, 0x4d, 0x61, 0x63, 0x68, 0x69, 0x6e, 0x65, 0x52, 0x07, 0x6d, 0x61,
	0x63, 0x68, 0x69, 0x6e, 0x65, 0x12, 0x2a, 0x0a, 0x06, 0x72, 0x65, 0x73, 0x75, 0x6c, 0x74, 0x18,
	0x02, 0x20, 0x01, 0x28, 0x0b, 0x32, 0x12, 0x2e, 0x67, 0x6f, 0x6f, 0x67, 0x6c, 0x65, 0x2e, 0x72,
	0x70, 0x63, 0x2e, 0x53, 0x74, 0x61, 0x74, 0x75, 0x73, 0x52, 0x06, 0x72, 0x65, 0x73, 0x75, 0x6c,
	0x74, 0x32, 0xb8, 0x01, 0x0a, 0x11, 0x53, 0x74, 0x61, 0x74, 0x65, 0x63, 0x68, 0x61, 0x72, 0x74,
	0x53, 0x65, 0x72, 0x76, 0x69, 0x63, 0x65, 0x12, 0x5e, 0x0a, 0x0d, 0x43, 0x72, 0x65, 0x61, 0x74,
	0x65, 0x4d, 0x61, 0x63, 0x68, 0x69, 0x6e, 0x65, 0x12, 0x24, 0x2e, 0x73, 0x74, 0x61, 0x74, 0x65,
	0x63, 0x68, 0x61, 0x72, 0x74, 0x73, 0x2e, 0x76, 0x31, 0x2e, 0x43, 0x72, 0x65, 0x61, 0x74, 0x65,
	0x4d, 0x61, 0x63, 0x68, 0x69, 0x6e, 0x65, 0x52, 0x65, 0x71, 0x75, 0x65, 0x73, 0x74, 0x1a, 0x25,
	0x2e, 0x73, 0x74, 0x61, 0x74, 0x65, 0x63, 0x68, 0x61, 0x72, 0x74, 0x73, 0x2e, 0x76, 0x31, 0x2e,
	0x43, 0x72, 0x65, 0x61, 0x74, 0x65, 0x4d, 0x61, 0x63, 0x68, 0x69, 0x6e, 0x65, 0x52, 0x65, 0x73,
	0x70, 0x6f, 0x6e, 0x73, 0x65, 0x22, 0x00, 0x12, 0x43, 0x0a, 0x04, 0x53, 0x74, 0x65, 0x70, 0x12,
	0x1b, 0x2e, 0x73, 0x74, 0x61, 0x74, 0x65, 0x63, 0x68, 0x61, 0x72, 0x74, 0x73, 0x2e, 0x76, 0x31,
	0x2e, 0x53, 0x74, 0x65, 0x70, 0x52, 0x65, 0x71, 0x75, 0x65, 0x73, 0x74, 0x1a, 0x1c, 0x2e, 0x73,
	0x74, 0x61, 0x74, 0x65, 0x63, 0x68, 0x61, 0x72, 0x74, 0x73, 0x2e, 0x76, 0x31, 0x2e, 0x53, 0x74,
	0x65, 0x70, 0x52, 0x65, 0x73, 0x70, 0x6f, 0x6e, 0x73, 0x65, 0x22, 0x00, 0x42, 0xb5, 0x01, 0x0a,
	0x12, 0x63, 0x6f, 0x6d, 0x2e, 0x73, 0x74, 0x61, 0x74, 0x65, 0x63, 0x68, 0x61, 0x72, 0x74, 0x73,
	0x2e, 0x76, 0x31, 0x42, 0x16, 0x53, 0x74, 0x61, 0x74, 0x65, 0x63, 0x68, 0x61, 0x72, 0x74, 0x53,
	0x65, 0x72, 0x76, 0x69, 0x63, 0x65, 0x50, 0x72, 0x6f, 0x74, 0x6f, 0x50, 0x01, 0x5a, 0x2e, 0x67,
	0x69, 0x74, 0x68, 0x75, 0x62, 0x2e, 0x63, 0x6f, 0x6d, 0x2f, 0x74, 0x6d, 0x63, 0x2f, 0x73, 0x63,
	0x2f, 0x73, 0x74, 0x61, 0x74, 0x65, 0x63, 0x68, 0x61, 0x72, 0x74, 0x73, 0x2f, 0x76, 0x31, 0x3b,
	0x73, 0x74, 0x61, 0x74, 0x65, 0x63, 0x68, 0x61, 0x72, 0x74, 0x73, 0x76, 0x31, 0xa2, 0x02, 0x03,
	0x53, 0x58, 0x58, 0xaa, 0x02, 0x0e, 0x53, 0x74, 0x61, 0x74, 0x65, 0x63, 0x68, 0x61, 0x72, 0x74,
	0x73, 0x2e, 0x56, 0x31, 0xca, 0x02, 0x0e, 0x53, 0x74, 0x61, 0x74, 0x65, 0x63, 0x68, 0x61, 0x72,
	0x74, 0x73, 0x5c, 0x56, 0x31, 0xe2, 0x02, 0x1a, 0x53, 0x74, 0x61, 0x74, 0x65, 0x63, 0x68, 0x61,
	0x72, 0x74, 0x73, 0x5c, 0x56, 0x31, 0x5c, 0x47, 0x50, 0x42, 0x4d, 0x65, 0x74, 0x61, 0x64, 0x61,
	0x74, 0x61, 0xea, 0x02, 0x0f, 0x53, 0x74, 0x61, 0x74, 0x65, 0x63, 0x68, 0x61, 0x72, 0x74, 0x73,
	0x3a, 0x3a, 0x56, 0x31, 0x62, 0x06, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x33,
}

var (
	file_statecharts_v1_statechart_service_proto_rawDescOnce sync.Once
	file_statecharts_v1_statechart_service_proto_rawDescData = file_statecharts_v1_statechart_service_proto_rawDesc
)

func file_statecharts_v1_statechart_service_proto_rawDescGZIP() []byte {
	file_statecharts_v1_statechart_service_proto_rawDescOnce.Do(func() {
		file_statecharts_v1_statechart_service_proto_rawDescData = protoimpl.X.CompressGZIP(file_statecharts_v1_statechart_service_proto_rawDescData)
	})
	return file_statecharts_v1_statechart_service_proto_rawDescData
}

var file_statecharts_v1_statechart_service_proto_msgTypes = make([]protoimpl.MessageInfo, 6)
var file_statecharts_v1_statechart_service_proto_goTypes = []interface{}{
	(*StatechartRegistry)(nil),    // 0: statecharts.v1.StatechartRegistry
	(*CreateMachineRequest)(nil),  // 1: statecharts.v1.CreateMachineRequest
	(*CreateMachineResponse)(nil), // 2: statecharts.v1.CreateMachineResponse
	(*StepRequest)(nil),           // 3: statecharts.v1.StepRequest
	(*StepResponse)(nil),          // 4: statecharts.v1.StepResponse
	nil,                           // 5: statecharts.v1.StatechartRegistry.StatechartsEntry
	(*structpb.Struct)(nil),       // 6: google.protobuf.Struct
	(*Machine)(nil),               // 7: statecharts.v1.Machine
	(*status.Status)(nil),         // 8: google.rpc.Status
	(*Statechart)(nil),            // 9: statecharts.v1.Statechart
}
var file_statecharts_v1_statechart_service_proto_depIdxs = []int32{
	5, // 0: statecharts.v1.StatechartRegistry.statecharts:type_name -> statecharts.v1.StatechartRegistry.StatechartsEntry
	6, // 1: statecharts.v1.CreateMachineRequest.context:type_name -> google.protobuf.Struct
	7, // 2: statecharts.v1.CreateMachineResponse.machine:type_name -> statecharts.v1.Machine
	7, // 3: statecharts.v1.StepResponse.machine:type_name -> statecharts.v1.Machine
	8, // 4: statecharts.v1.StepResponse.result:type_name -> google.rpc.Status
	9, // 5: statecharts.v1.StatechartRegistry.StatechartsEntry.value:type_name -> statecharts.v1.Statechart
	1, // 6: statecharts.v1.StatechartService.CreateMachine:input_type -> statecharts.v1.CreateMachineRequest
	3, // 7: statecharts.v1.StatechartService.Step:input_type -> statecharts.v1.StepRequest
	2, // 8: statecharts.v1.StatechartService.CreateMachine:output_type -> statecharts.v1.CreateMachineResponse
	4, // 9: statecharts.v1.StatechartService.Step:output_type -> statecharts.v1.StepResponse
	8, // [8:10] is the sub-list for method output_type
	6, // [6:8] is the sub-list for method input_type
	6, // [6:6] is the sub-list for extension type_name
	6, // [6:6] is the sub-list for extension extendee
	0, // [0:6] is the sub-list for field type_name
}

func init() { file_statecharts_v1_statechart_service_proto_init() }
func file_statecharts_v1_statechart_service_proto_init() {
	if File_statecharts_v1_statechart_service_proto != nil {
		return
	}
	file_statecharts_v1_statecharts_proto_init()
	if !protoimpl.UnsafeEnabled {
		file_statecharts_v1_statechart_service_proto_msgTypes[0].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*StatechartRegistry); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_statecharts_v1_statechart_service_proto_msgTypes[1].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*CreateMachineRequest); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_statecharts_v1_statechart_service_proto_msgTypes[2].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*CreateMachineResponse); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_statecharts_v1_statechart_service_proto_msgTypes[3].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*StepRequest); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_statecharts_v1_statechart_service_proto_msgTypes[4].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*StepResponse); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
	}
	type x struct{}
	out := protoimpl.TypeBuilder{
		File: protoimpl.DescBuilder{
			GoPackagePath: reflect.TypeOf(x{}).PkgPath(),
			RawDescriptor: file_statecharts_v1_statechart_service_proto_rawDesc,
			NumEnums:      0,
			NumMessages:   6,
			NumExtensions: 0,
			NumServices:   1,
		},
		GoTypes:           file_statecharts_v1_statechart_service_proto_goTypes,
		DependencyIndexes: file_statecharts_v1_statechart_service_proto_depIdxs,
		MessageInfos:      file_statecharts_v1_statechart_service_proto_msgTypes,
	}.Build()
	File_statecharts_v1_statechart_service_proto = out.File
	file_statecharts_v1_statechart_service_proto_rawDesc = nil
	file_statecharts_v1_statechart_service_proto_goTypes = nil
	file_statecharts_v1_statechart_service_proto_depIdxs = nil
}
