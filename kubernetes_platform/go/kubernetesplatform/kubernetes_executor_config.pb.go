// Copyright 2023 The Kubeflow Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Code generated by protoc-gen-go. DO NOT EDIT.
// versions:
// 	protoc-gen-go v1.26.0
// 	protoc        v3.17.3
// source: kubernetes_executor_config.proto

package kubernetesplatform

import (
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

type KubernetesExecutorConfig struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	SecretAsVolume []*SecretAsVolume `protobuf:"bytes,1,rep,name=secret_as_volume,json=secretAsVolume,proto3" json:"secret_as_volume,omitempty"`
	SecretAsEnv    []*SecretAsEnv    `protobuf:"bytes,2,rep,name=secret_as_env,json=secretAsEnv,proto3" json:"secret_as_env,omitempty"`
	PvcMount       []*PvcMount       `protobuf:"bytes,3,rep,name=pvc_mount,json=pvcMount,proto3" json:"pvc_mount,omitempty"`
	NodeSelector   []*NodeSelector   `protobuf:"bytes,4,rep,name=node_selector,json=nodeSelector,proto3" json:"node_selector,omitempty"`
}

func (x *KubernetesExecutorConfig) Reset() {
	*x = KubernetesExecutorConfig{}
	if protoimpl.UnsafeEnabled {
		mi := &file_kubernetes_executor_config_proto_msgTypes[0]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *KubernetesExecutorConfig) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*KubernetesExecutorConfig) ProtoMessage() {}

func (x *KubernetesExecutorConfig) ProtoReflect() protoreflect.Message {
	mi := &file_kubernetes_executor_config_proto_msgTypes[0]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use KubernetesExecutorConfig.ProtoReflect.Descriptor instead.
func (*KubernetesExecutorConfig) Descriptor() ([]byte, []int) {
	return file_kubernetes_executor_config_proto_rawDescGZIP(), []int{0}
}

func (x *KubernetesExecutorConfig) GetSecretAsVolume() []*SecretAsVolume {
	if x != nil {
		return x.SecretAsVolume
	}
	return nil
}

func (x *KubernetesExecutorConfig) GetSecretAsEnv() []*SecretAsEnv {
	if x != nil {
		return x.SecretAsEnv
	}
	return nil
}

func (x *KubernetesExecutorConfig) GetPvcMount() []*PvcMount {
	if x != nil {
		return x.PvcMount
	}
	return nil
}

func (x *KubernetesExecutorConfig) GetNodeSelector() []*NodeSelector {
	if x != nil {
		return x.NodeSelector
	}
	return nil
}

type SecretAsVolume struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	// Name of the Secret.
	SecretName string `protobuf:"bytes,1,opt,name=secret_name,json=secretName,proto3" json:"secret_name,omitempty"`
	// Container path to mount the Secret data.
	MountPath string `protobuf:"bytes,2,opt,name=mount_path,json=mountPath,proto3" json:"mount_path,omitempty"`
}

func (x *SecretAsVolume) Reset() {
	*x = SecretAsVolume{}
	if protoimpl.UnsafeEnabled {
		mi := &file_kubernetes_executor_config_proto_msgTypes[1]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *SecretAsVolume) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*SecretAsVolume) ProtoMessage() {}

func (x *SecretAsVolume) ProtoReflect() protoreflect.Message {
	mi := &file_kubernetes_executor_config_proto_msgTypes[1]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use SecretAsVolume.ProtoReflect.Descriptor instead.
func (*SecretAsVolume) Descriptor() ([]byte, []int) {
	return file_kubernetes_executor_config_proto_rawDescGZIP(), []int{1}
}

func (x *SecretAsVolume) GetSecretName() string {
	if x != nil {
		return x.SecretName
	}
	return ""
}

func (x *SecretAsVolume) GetMountPath() string {
	if x != nil {
		return x.MountPath
	}
	return ""
}

type SecretAsEnv struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	// Name of the Secret.
	SecretName string                           `protobuf:"bytes,1,opt,name=secret_name,json=secretName,proto3" json:"secret_name,omitempty"`
	KeyToEnv   []*SecretAsEnv_SecretKeyToEnvMap `protobuf:"bytes,2,rep,name=key_to_env,json=keyToEnv,proto3" json:"key_to_env,omitempty"`
}

func (x *SecretAsEnv) Reset() {
	*x = SecretAsEnv{}
	if protoimpl.UnsafeEnabled {
		mi := &file_kubernetes_executor_config_proto_msgTypes[2]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *SecretAsEnv) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*SecretAsEnv) ProtoMessage() {}

func (x *SecretAsEnv) ProtoReflect() protoreflect.Message {
	mi := &file_kubernetes_executor_config_proto_msgTypes[2]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use SecretAsEnv.ProtoReflect.Descriptor instead.
func (*SecretAsEnv) Descriptor() ([]byte, []int) {
	return file_kubernetes_executor_config_proto_rawDescGZIP(), []int{2}
}

func (x *SecretAsEnv) GetSecretName() string {
	if x != nil {
		return x.SecretName
	}
	return ""
}

func (x *SecretAsEnv) GetKeyToEnv() []*SecretAsEnv_SecretKeyToEnvMap {
	if x != nil {
		return x.KeyToEnv
	}
	return nil
}

// Represents an upstream task's output parameter.
type TaskOutputParameterSpec struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	// The name of the upstream task which produces the output parameter that
	// matches with the `output_parameter_key`.
	ProducerTask string `protobuf:"bytes,1,opt,name=producer_task,json=producerTask,proto3" json:"producer_task,omitempty"`
	// The key of [TaskOutputsSpec.parameters][] map of the producer task.
	OutputParameterKey string `protobuf:"bytes,2,opt,name=output_parameter_key,json=outputParameterKey,proto3" json:"output_parameter_key,omitempty"`
}

func (x *TaskOutputParameterSpec) Reset() {
	*x = TaskOutputParameterSpec{}
	if protoimpl.UnsafeEnabled {
		mi := &file_kubernetes_executor_config_proto_msgTypes[3]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *TaskOutputParameterSpec) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*TaskOutputParameterSpec) ProtoMessage() {}

func (x *TaskOutputParameterSpec) ProtoReflect() protoreflect.Message {
	mi := &file_kubernetes_executor_config_proto_msgTypes[3]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use TaskOutputParameterSpec.ProtoReflect.Descriptor instead.
func (*TaskOutputParameterSpec) Descriptor() ([]byte, []int) {
	return file_kubernetes_executor_config_proto_rawDescGZIP(), []int{3}
}

func (x *TaskOutputParameterSpec) GetProducerTask() string {
	if x != nil {
		return x.ProducerTask
	}
	return ""
}

func (x *TaskOutputParameterSpec) GetOutputParameterKey() string {
	if x != nil {
		return x.OutputParameterKey
	}
	return ""
}

type PvcMount struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	// Identifier for the PVC.
	// Used like TaskInputsSpec.InputParameterSpec.kind.
	//
	// Types that are assignable to PvcReference:
	//	*PvcMount_TaskOutputParameter
	//	*PvcMount_Constant
	//	*PvcMount_ComponentInputParameter
	PvcReference isPvcMount_PvcReference `protobuf_oneof:"pvc_reference"`
	// Container path to which the PVC should be mounted.
	MountPath string `protobuf:"bytes,4,opt,name=mount_path,json=mountPath,proto3" json:"mount_path,omitempty"`
}

func (x *PvcMount) Reset() {
	*x = PvcMount{}
	if protoimpl.UnsafeEnabled {
		mi := &file_kubernetes_executor_config_proto_msgTypes[4]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *PvcMount) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*PvcMount) ProtoMessage() {}

func (x *PvcMount) ProtoReflect() protoreflect.Message {
	mi := &file_kubernetes_executor_config_proto_msgTypes[4]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use PvcMount.ProtoReflect.Descriptor instead.
func (*PvcMount) Descriptor() ([]byte, []int) {
	return file_kubernetes_executor_config_proto_rawDescGZIP(), []int{4}
}

func (m *PvcMount) GetPvcReference() isPvcMount_PvcReference {
	if m != nil {
		return m.PvcReference
	}
	return nil
}

func (x *PvcMount) GetTaskOutputParameter() *TaskOutputParameterSpec {
	if x, ok := x.GetPvcReference().(*PvcMount_TaskOutputParameter); ok {
		return x.TaskOutputParameter
	}
	return nil
}

func (x *PvcMount) GetConstant() string {
	if x, ok := x.GetPvcReference().(*PvcMount_Constant); ok {
		return x.Constant
	}
	return ""
}

func (x *PvcMount) GetComponentInputParameter() string {
	if x, ok := x.GetPvcReference().(*PvcMount_ComponentInputParameter); ok {
		return x.ComponentInputParameter
	}
	return ""
}

func (x *PvcMount) GetMountPath() string {
	if x != nil {
		return x.MountPath
	}
	return ""
}

type isPvcMount_PvcReference interface {
	isPvcMount_PvcReference()
}

type PvcMount_TaskOutputParameter struct {
	// Output parameter from an upstream task.
	TaskOutputParameter *TaskOutputParameterSpec `protobuf:"bytes,1,opt,name=task_output_parameter,json=taskOutputParameter,proto3,oneof"`
}

type PvcMount_Constant struct {
	// A constant value.
	Constant string `protobuf:"bytes,2,opt,name=constant,proto3,oneof"`
}

type PvcMount_ComponentInputParameter struct {
	// Pass the input parameter from parent component input parameter.
	ComponentInputParameter string `protobuf:"bytes,3,opt,name=component_input_parameter,json=componentInputParameter,proto3,oneof"`
}

func (*PvcMount_TaskOutputParameter) isPvcMount_PvcReference() {}

func (*PvcMount_Constant) isPvcMount_PvcReference() {}

func (*PvcMount_ComponentInputParameter) isPvcMount_PvcReference() {}

type CreatePvc struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	// Types that are assignable to Name:
	//	*CreatePvc_PvcName
	//	*CreatePvc_PvcNameSuffix
	Name isCreatePvc_Name `protobuf_oneof:"name"`
	// Corresponds to PersistentVolumeClaim.spec.accessMode field.
	AccessModes []string `protobuf:"bytes,3,rep,name=access_modes,json=accessModes,proto3" json:"access_modes,omitempty"`
	// Corresponds to PersistentVolumeClaim.spec.resources.requests.storage field.
	Size string `protobuf:"bytes,4,opt,name=size,proto3" json:"size,omitempty"`
	// If true, corresponds to omitted PersistentVolumeClaim.spec.storageClassName.
	DefaultStorageClass bool `protobuf:"varint,5,opt,name=default_storage_class,json=defaultStorageClass,proto3" json:"default_storage_class,omitempty"`
	// Corresponds to PersistentVolumeClaim.spec.storageClassName string field.
	// Should only be used if default_storage_class is false.
	StorageClassName string `protobuf:"bytes,6,opt,name=storage_class_name,json=storageClassName,proto3" json:"storage_class_name,omitempty"`
	// Corresponds to PersistentVolumeClaim.spec.volumeName field.
	VolumeName string `protobuf:"bytes,7,opt,name=volume_name,json=volumeName,proto3" json:"volume_name,omitempty"`
	// Corresponds to PersistentVolumeClaim.metadata.annotations field.
	Annotations *structpb.Struct `protobuf:"bytes,8,opt,name=annotations,proto3" json:"annotations,omitempty"`
}

func (x *CreatePvc) Reset() {
	*x = CreatePvc{}
	if protoimpl.UnsafeEnabled {
		mi := &file_kubernetes_executor_config_proto_msgTypes[5]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *CreatePvc) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*CreatePvc) ProtoMessage() {}

func (x *CreatePvc) ProtoReflect() protoreflect.Message {
	mi := &file_kubernetes_executor_config_proto_msgTypes[5]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use CreatePvc.ProtoReflect.Descriptor instead.
func (*CreatePvc) Descriptor() ([]byte, []int) {
	return file_kubernetes_executor_config_proto_rawDescGZIP(), []int{5}
}

func (m *CreatePvc) GetName() isCreatePvc_Name {
	if m != nil {
		return m.Name
	}
	return nil
}

func (x *CreatePvc) GetPvcName() string {
	if x, ok := x.GetName().(*CreatePvc_PvcName); ok {
		return x.PvcName
	}
	return ""
}

func (x *CreatePvc) GetPvcNameSuffix() string {
	if x, ok := x.GetName().(*CreatePvc_PvcNameSuffix); ok {
		return x.PvcNameSuffix
	}
	return ""
}

func (x *CreatePvc) GetAccessModes() []string {
	if x != nil {
		return x.AccessModes
	}
	return nil
}

func (x *CreatePvc) GetSize() string {
	if x != nil {
		return x.Size
	}
	return ""
}

func (x *CreatePvc) GetDefaultStorageClass() bool {
	if x != nil {
		return x.DefaultStorageClass
	}
	return false
}

func (x *CreatePvc) GetStorageClassName() string {
	if x != nil {
		return x.StorageClassName
	}
	return ""
}

func (x *CreatePvc) GetVolumeName() string {
	if x != nil {
		return x.VolumeName
	}
	return ""
}

func (x *CreatePvc) GetAnnotations() *structpb.Struct {
	if x != nil {
		return x.Annotations
	}
	return nil
}

type isCreatePvc_Name interface {
	isCreatePvc_Name()
}

type CreatePvc_PvcName struct {
	// Name of the PVC, if not dynamically generated.
	PvcName string `protobuf:"bytes,1,opt,name=pvc_name,json=pvcName,proto3,oneof"`
}

type CreatePvc_PvcNameSuffix struct {
	// Suffix for a dynamically generated PVC name of the form
	// {{workflow.name}}-<pvc_name_suffix>.
	PvcNameSuffix string `protobuf:"bytes,2,opt,name=pvc_name_suffix,json=pvcNameSuffix,proto3,oneof"`
}

func (*CreatePvc_PvcName) isCreatePvc_Name() {}

func (*CreatePvc_PvcNameSuffix) isCreatePvc_Name() {}

type DeletePvc struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	// Identifier for the PVC.
	// Used like TaskInputsSpec.InputParameterSpec.kind.
	//
	// Types that are assignable to PvcReference:
	//	*DeletePvc_TaskOutputParameter
	//	*DeletePvc_Constant
	//	*DeletePvc_ComponentInputParameter
	PvcReference isDeletePvc_PvcReference `protobuf_oneof:"pvc_reference"`
}

func (x *DeletePvc) Reset() {
	*x = DeletePvc{}
	if protoimpl.UnsafeEnabled {
		mi := &file_kubernetes_executor_config_proto_msgTypes[6]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *DeletePvc) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*DeletePvc) ProtoMessage() {}

func (x *DeletePvc) ProtoReflect() protoreflect.Message {
	mi := &file_kubernetes_executor_config_proto_msgTypes[6]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use DeletePvc.ProtoReflect.Descriptor instead.
func (*DeletePvc) Descriptor() ([]byte, []int) {
	return file_kubernetes_executor_config_proto_rawDescGZIP(), []int{6}
}

func (m *DeletePvc) GetPvcReference() isDeletePvc_PvcReference {
	if m != nil {
		return m.PvcReference
	}
	return nil
}

func (x *DeletePvc) GetTaskOutputParameter() *TaskOutputParameterSpec {
	if x, ok := x.GetPvcReference().(*DeletePvc_TaskOutputParameter); ok {
		return x.TaskOutputParameter
	}
	return nil
}

func (x *DeletePvc) GetConstant() string {
	if x, ok := x.GetPvcReference().(*DeletePvc_Constant); ok {
		return x.Constant
	}
	return ""
}

func (x *DeletePvc) GetComponentInputParameter() string {
	if x, ok := x.GetPvcReference().(*DeletePvc_ComponentInputParameter); ok {
		return x.ComponentInputParameter
	}
	return ""
}

type isDeletePvc_PvcReference interface {
	isDeletePvc_PvcReference()
}

type DeletePvc_TaskOutputParameter struct {
	// Output parameter from an upstream task.
	TaskOutputParameter *TaskOutputParameterSpec `protobuf:"bytes,1,opt,name=task_output_parameter,json=taskOutputParameter,proto3,oneof"`
}

type DeletePvc_Constant struct {
	// A constant value.
	Constant string `protobuf:"bytes,2,opt,name=constant,proto3,oneof"`
}

type DeletePvc_ComponentInputParameter struct {
	// Pass the input parameter from parent component input parameter.
	ComponentInputParameter string `protobuf:"bytes,3,opt,name=component_input_parameter,json=componentInputParameter,proto3,oneof"`
}

func (*DeletePvc_TaskOutputParameter) isDeletePvc_PvcReference() {}

func (*DeletePvc_Constant) isDeletePvc_PvcReference() {}

func (*DeletePvc_ComponentInputParameter) isDeletePvc_PvcReference() {}

type NodeSelector struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	LabelKey   string `protobuf:"bytes,1,opt,name=label_key,json=labelKey,proto3" json:"label_key,omitempty"`
	LabelValue string `protobuf:"bytes,2,opt,name=label_value,json=labelValue,proto3" json:"label_value,omitempty"`
}

func (x *NodeSelector) Reset() {
	*x = NodeSelector{}
	if protoimpl.UnsafeEnabled {
		mi := &file_kubernetes_executor_config_proto_msgTypes[7]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *NodeSelector) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*NodeSelector) ProtoMessage() {}

func (x *NodeSelector) ProtoReflect() protoreflect.Message {
	mi := &file_kubernetes_executor_config_proto_msgTypes[7]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use NodeSelector.ProtoReflect.Descriptor instead.
func (*NodeSelector) Descriptor() ([]byte, []int) {
	return file_kubernetes_executor_config_proto_rawDescGZIP(), []int{7}
}

func (x *NodeSelector) GetLabelKey() string {
	if x != nil {
		return x.LabelKey
	}
	return ""
}

func (x *NodeSelector) GetLabelValue() string {
	if x != nil {
		return x.LabelValue
	}
	return ""
}

type SecretAsEnv_SecretKeyToEnvMap struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	// Corresponds to a key of the Secret.data field.
	SecretKey string `protobuf:"bytes,1,opt,name=secret_key,json=secretKey,proto3" json:"secret_key,omitempty"`
	// Env var to which secret_key's data should be set.
	EnvVar string `protobuf:"bytes,2,opt,name=env_var,json=envVar,proto3" json:"env_var,omitempty"`
}

func (x *SecretAsEnv_SecretKeyToEnvMap) Reset() {
	*x = SecretAsEnv_SecretKeyToEnvMap{}
	if protoimpl.UnsafeEnabled {
		mi := &file_kubernetes_executor_config_proto_msgTypes[8]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *SecretAsEnv_SecretKeyToEnvMap) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*SecretAsEnv_SecretKeyToEnvMap) ProtoMessage() {}

func (x *SecretAsEnv_SecretKeyToEnvMap) ProtoReflect() protoreflect.Message {
	mi := &file_kubernetes_executor_config_proto_msgTypes[8]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use SecretAsEnv_SecretKeyToEnvMap.ProtoReflect.Descriptor instead.
func (*SecretAsEnv_SecretKeyToEnvMap) Descriptor() ([]byte, []int) {
	return file_kubernetes_executor_config_proto_rawDescGZIP(), []int{2, 0}
}

func (x *SecretAsEnv_SecretKeyToEnvMap) GetSecretKey() string {
	if x != nil {
		return x.SecretKey
	}
	return ""
}

func (x *SecretAsEnv_SecretKeyToEnvMap) GetEnvVar() string {
	if x != nil {
		return x.EnvVar
	}
	return ""
}

var File_kubernetes_executor_config_proto protoreflect.FileDescriptor

var file_kubernetes_executor_config_proto_rawDesc = []byte{
	0x0a, 0x20, 0x6b, 0x75, 0x62, 0x65, 0x72, 0x6e, 0x65, 0x74, 0x65, 0x73, 0x5f, 0x65, 0x78, 0x65,
	0x63, 0x75, 0x74, 0x6f, 0x72, 0x5f, 0x63, 0x6f, 0x6e, 0x66, 0x69, 0x67, 0x2e, 0x70, 0x72, 0x6f,
	0x74, 0x6f, 0x12, 0x0e, 0x6b, 0x66, 0x70, 0x5f, 0x6b, 0x75, 0x62, 0x65, 0x72, 0x6e, 0x65, 0x74,
	0x65, 0x73, 0x1a, 0x1c, 0x67, 0x6f, 0x6f, 0x67, 0x6c, 0x65, 0x2f, 0x70, 0x72, 0x6f, 0x74, 0x6f,
	0x62, 0x75, 0x66, 0x2f, 0x73, 0x74, 0x72, 0x75, 0x63, 0x74, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f,
	0x22, 0x9f, 0x02, 0x0a, 0x18, 0x4b, 0x75, 0x62, 0x65, 0x72, 0x6e, 0x65, 0x74, 0x65, 0x73, 0x45,
	0x78, 0x65, 0x63, 0x75, 0x74, 0x6f, 0x72, 0x43, 0x6f, 0x6e, 0x66, 0x69, 0x67, 0x12, 0x48, 0x0a,
	0x10, 0x73, 0x65, 0x63, 0x72, 0x65, 0x74, 0x5f, 0x61, 0x73, 0x5f, 0x76, 0x6f, 0x6c, 0x75, 0x6d,
	0x65, 0x18, 0x01, 0x20, 0x03, 0x28, 0x0b, 0x32, 0x1e, 0x2e, 0x6b, 0x66, 0x70, 0x5f, 0x6b, 0x75,
	0x62, 0x65, 0x72, 0x6e, 0x65, 0x74, 0x65, 0x73, 0x2e, 0x53, 0x65, 0x63, 0x72, 0x65, 0x74, 0x41,
	0x73, 0x56, 0x6f, 0x6c, 0x75, 0x6d, 0x65, 0x52, 0x0e, 0x73, 0x65, 0x63, 0x72, 0x65, 0x74, 0x41,
	0x73, 0x56, 0x6f, 0x6c, 0x75, 0x6d, 0x65, 0x12, 0x3f, 0x0a, 0x0d, 0x73, 0x65, 0x63, 0x72, 0x65,
	0x74, 0x5f, 0x61, 0x73, 0x5f, 0x65, 0x6e, 0x76, 0x18, 0x02, 0x20, 0x03, 0x28, 0x0b, 0x32, 0x1b,
	0x2e, 0x6b, 0x66, 0x70, 0x5f, 0x6b, 0x75, 0x62, 0x65, 0x72, 0x6e, 0x65, 0x74, 0x65, 0x73, 0x2e,
	0x53, 0x65, 0x63, 0x72, 0x65, 0x74, 0x41, 0x73, 0x45, 0x6e, 0x76, 0x52, 0x0b, 0x73, 0x65, 0x63,
	0x72, 0x65, 0x74, 0x41, 0x73, 0x45, 0x6e, 0x76, 0x12, 0x35, 0x0a, 0x09, 0x70, 0x76, 0x63, 0x5f,
	0x6d, 0x6f, 0x75, 0x6e, 0x74, 0x18, 0x03, 0x20, 0x03, 0x28, 0x0b, 0x32, 0x18, 0x2e, 0x6b, 0x66,
	0x70, 0x5f, 0x6b, 0x75, 0x62, 0x65, 0x72, 0x6e, 0x65, 0x74, 0x65, 0x73, 0x2e, 0x50, 0x76, 0x63,
	0x4d, 0x6f, 0x75, 0x6e, 0x74, 0x52, 0x08, 0x70, 0x76, 0x63, 0x4d, 0x6f, 0x75, 0x6e, 0x74, 0x12,
	0x41, 0x0a, 0x0d, 0x6e, 0x6f, 0x64, 0x65, 0x5f, 0x73, 0x65, 0x6c, 0x65, 0x63, 0x74, 0x6f, 0x72,
	0x18, 0x04, 0x20, 0x03, 0x28, 0x0b, 0x32, 0x1c, 0x2e, 0x6b, 0x66, 0x70, 0x5f, 0x6b, 0x75, 0x62,
	0x65, 0x72, 0x6e, 0x65, 0x74, 0x65, 0x73, 0x2e, 0x4e, 0x6f, 0x64, 0x65, 0x53, 0x65, 0x6c, 0x65,
	0x63, 0x74, 0x6f, 0x72, 0x52, 0x0c, 0x6e, 0x6f, 0x64, 0x65, 0x53, 0x65, 0x6c, 0x65, 0x63, 0x74,
	0x6f, 0x72, 0x22, 0x50, 0x0a, 0x0e, 0x53, 0x65, 0x63, 0x72, 0x65, 0x74, 0x41, 0x73, 0x56, 0x6f,
	0x6c, 0x75, 0x6d, 0x65, 0x12, 0x1f, 0x0a, 0x0b, 0x73, 0x65, 0x63, 0x72, 0x65, 0x74, 0x5f, 0x6e,
	0x61, 0x6d, 0x65, 0x18, 0x01, 0x20, 0x01, 0x28, 0x09, 0x52, 0x0a, 0x73, 0x65, 0x63, 0x72, 0x65,
	0x74, 0x4e, 0x61, 0x6d, 0x65, 0x12, 0x1d, 0x0a, 0x0a, 0x6d, 0x6f, 0x75, 0x6e, 0x74, 0x5f, 0x70,
	0x61, 0x74, 0x68, 0x18, 0x02, 0x20, 0x01, 0x28, 0x09, 0x52, 0x09, 0x6d, 0x6f, 0x75, 0x6e, 0x74,
	0x50, 0x61, 0x74, 0x68, 0x22, 0xc8, 0x01, 0x0a, 0x0b, 0x53, 0x65, 0x63, 0x72, 0x65, 0x74, 0x41,
	0x73, 0x45, 0x6e, 0x76, 0x12, 0x1f, 0x0a, 0x0b, 0x73, 0x65, 0x63, 0x72, 0x65, 0x74, 0x5f, 0x6e,
	0x61, 0x6d, 0x65, 0x18, 0x01, 0x20, 0x01, 0x28, 0x09, 0x52, 0x0a, 0x73, 0x65, 0x63, 0x72, 0x65,
	0x74, 0x4e, 0x61, 0x6d, 0x65, 0x12, 0x4b, 0x0a, 0x0a, 0x6b, 0x65, 0x79, 0x5f, 0x74, 0x6f, 0x5f,
	0x65, 0x6e, 0x76, 0x18, 0x02, 0x20, 0x03, 0x28, 0x0b, 0x32, 0x2d, 0x2e, 0x6b, 0x66, 0x70, 0x5f,
	0x6b, 0x75, 0x62, 0x65, 0x72, 0x6e, 0x65, 0x74, 0x65, 0x73, 0x2e, 0x53, 0x65, 0x63, 0x72, 0x65,
	0x74, 0x41, 0x73, 0x45, 0x6e, 0x76, 0x2e, 0x53, 0x65, 0x63, 0x72, 0x65, 0x74, 0x4b, 0x65, 0x79,
	0x54, 0x6f, 0x45, 0x6e, 0x76, 0x4d, 0x61, 0x70, 0x52, 0x08, 0x6b, 0x65, 0x79, 0x54, 0x6f, 0x45,
	0x6e, 0x76, 0x1a, 0x4b, 0x0a, 0x11, 0x53, 0x65, 0x63, 0x72, 0x65, 0x74, 0x4b, 0x65, 0x79, 0x54,
	0x6f, 0x45, 0x6e, 0x76, 0x4d, 0x61, 0x70, 0x12, 0x1d, 0x0a, 0x0a, 0x73, 0x65, 0x63, 0x72, 0x65,
	0x74, 0x5f, 0x6b, 0x65, 0x79, 0x18, 0x01, 0x20, 0x01, 0x28, 0x09, 0x52, 0x09, 0x73, 0x65, 0x63,
	0x72, 0x65, 0x74, 0x4b, 0x65, 0x79, 0x12, 0x17, 0x0a, 0x07, 0x65, 0x6e, 0x76, 0x5f, 0x76, 0x61,
	0x72, 0x18, 0x02, 0x20, 0x01, 0x28, 0x09, 0x52, 0x06, 0x65, 0x6e, 0x76, 0x56, 0x61, 0x72, 0x22,
	0x70, 0x0a, 0x17, 0x54, 0x61, 0x73, 0x6b, 0x4f, 0x75, 0x74, 0x70, 0x75, 0x74, 0x50, 0x61, 0x72,
	0x61, 0x6d, 0x65, 0x74, 0x65, 0x72, 0x53, 0x70, 0x65, 0x63, 0x12, 0x23, 0x0a, 0x0d, 0x70, 0x72,
	0x6f, 0x64, 0x75, 0x63, 0x65, 0x72, 0x5f, 0x74, 0x61, 0x73, 0x6b, 0x18, 0x01, 0x20, 0x01, 0x28,
	0x09, 0x52, 0x0c, 0x70, 0x72, 0x6f, 0x64, 0x75, 0x63, 0x65, 0x72, 0x54, 0x61, 0x73, 0x6b, 0x12,
	0x30, 0x0a, 0x14, 0x6f, 0x75, 0x74, 0x70, 0x75, 0x74, 0x5f, 0x70, 0x61, 0x72, 0x61, 0x6d, 0x65,
	0x74, 0x65, 0x72, 0x5f, 0x6b, 0x65, 0x79, 0x18, 0x02, 0x20, 0x01, 0x28, 0x09, 0x52, 0x12, 0x6f,
	0x75, 0x74, 0x70, 0x75, 0x74, 0x50, 0x61, 0x72, 0x61, 0x6d, 0x65, 0x74, 0x65, 0x72, 0x4b, 0x65,
	0x79, 0x22, 0xf5, 0x01, 0x0a, 0x08, 0x50, 0x76, 0x63, 0x4d, 0x6f, 0x75, 0x6e, 0x74, 0x12, 0x5d,
	0x0a, 0x15, 0x74, 0x61, 0x73, 0x6b, 0x5f, 0x6f, 0x75, 0x74, 0x70, 0x75, 0x74, 0x5f, 0x70, 0x61,
	0x72, 0x61, 0x6d, 0x65, 0x74, 0x65, 0x72, 0x18, 0x01, 0x20, 0x01, 0x28, 0x0b, 0x32, 0x27, 0x2e,
	0x6b, 0x66, 0x70, 0x5f, 0x6b, 0x75, 0x62, 0x65, 0x72, 0x6e, 0x65, 0x74, 0x65, 0x73, 0x2e, 0x54,
	0x61, 0x73, 0x6b, 0x4f, 0x75, 0x74, 0x70, 0x75, 0x74, 0x50, 0x61, 0x72, 0x61, 0x6d, 0x65, 0x74,
	0x65, 0x72, 0x53, 0x70, 0x65, 0x63, 0x48, 0x00, 0x52, 0x13, 0x74, 0x61, 0x73, 0x6b, 0x4f, 0x75,
	0x74, 0x70, 0x75, 0x74, 0x50, 0x61, 0x72, 0x61, 0x6d, 0x65, 0x74, 0x65, 0x72, 0x12, 0x1c, 0x0a,
	0x08, 0x63, 0x6f, 0x6e, 0x73, 0x74, 0x61, 0x6e, 0x74, 0x18, 0x02, 0x20, 0x01, 0x28, 0x09, 0x48,
	0x00, 0x52, 0x08, 0x63, 0x6f, 0x6e, 0x73, 0x74, 0x61, 0x6e, 0x74, 0x12, 0x3c, 0x0a, 0x19, 0x63,
	0x6f, 0x6d, 0x70, 0x6f, 0x6e, 0x65, 0x6e, 0x74, 0x5f, 0x69, 0x6e, 0x70, 0x75, 0x74, 0x5f, 0x70,
	0x61, 0x72, 0x61, 0x6d, 0x65, 0x74, 0x65, 0x72, 0x18, 0x03, 0x20, 0x01, 0x28, 0x09, 0x48, 0x00,
	0x52, 0x17, 0x63, 0x6f, 0x6d, 0x70, 0x6f, 0x6e, 0x65, 0x6e, 0x74, 0x49, 0x6e, 0x70, 0x75, 0x74,
	0x50, 0x61, 0x72, 0x61, 0x6d, 0x65, 0x74, 0x65, 0x72, 0x12, 0x1d, 0x0a, 0x0a, 0x6d, 0x6f, 0x75,
	0x6e, 0x74, 0x5f, 0x70, 0x61, 0x74, 0x68, 0x18, 0x04, 0x20, 0x01, 0x28, 0x09, 0x52, 0x09, 0x6d,
	0x6f, 0x75, 0x6e, 0x74, 0x50, 0x61, 0x74, 0x68, 0x42, 0x0f, 0x0a, 0x0d, 0x70, 0x76, 0x63, 0x5f,
	0x72, 0x65, 0x66, 0x65, 0x72, 0x65, 0x6e, 0x63, 0x65, 0x22, 0xcf, 0x02, 0x0a, 0x09, 0x43, 0x72,
	0x65, 0x61, 0x74, 0x65, 0x50, 0x76, 0x63, 0x12, 0x1b, 0x0a, 0x08, 0x70, 0x76, 0x63, 0x5f, 0x6e,
	0x61, 0x6d, 0x65, 0x18, 0x01, 0x20, 0x01, 0x28, 0x09, 0x48, 0x00, 0x52, 0x07, 0x70, 0x76, 0x63,
	0x4e, 0x61, 0x6d, 0x65, 0x12, 0x28, 0x0a, 0x0f, 0x70, 0x76, 0x63, 0x5f, 0x6e, 0x61, 0x6d, 0x65,
	0x5f, 0x73, 0x75, 0x66, 0x66, 0x69, 0x78, 0x18, 0x02, 0x20, 0x01, 0x28, 0x09, 0x48, 0x00, 0x52,
	0x0d, 0x70, 0x76, 0x63, 0x4e, 0x61, 0x6d, 0x65, 0x53, 0x75, 0x66, 0x66, 0x69, 0x78, 0x12, 0x21,
	0x0a, 0x0c, 0x61, 0x63, 0x63, 0x65, 0x73, 0x73, 0x5f, 0x6d, 0x6f, 0x64, 0x65, 0x73, 0x18, 0x03,
	0x20, 0x03, 0x28, 0x09, 0x52, 0x0b, 0x61, 0x63, 0x63, 0x65, 0x73, 0x73, 0x4d, 0x6f, 0x64, 0x65,
	0x73, 0x12, 0x12, 0x0a, 0x04, 0x73, 0x69, 0x7a, 0x65, 0x18, 0x04, 0x20, 0x01, 0x28, 0x09, 0x52,
	0x04, 0x73, 0x69, 0x7a, 0x65, 0x12, 0x32, 0x0a, 0x15, 0x64, 0x65, 0x66, 0x61, 0x75, 0x6c, 0x74,
	0x5f, 0x73, 0x74, 0x6f, 0x72, 0x61, 0x67, 0x65, 0x5f, 0x63, 0x6c, 0x61, 0x73, 0x73, 0x18, 0x05,
	0x20, 0x01, 0x28, 0x08, 0x52, 0x13, 0x64, 0x65, 0x66, 0x61, 0x75, 0x6c, 0x74, 0x53, 0x74, 0x6f,
	0x72, 0x61, 0x67, 0x65, 0x43, 0x6c, 0x61, 0x73, 0x73, 0x12, 0x2c, 0x0a, 0x12, 0x73, 0x74, 0x6f,
	0x72, 0x61, 0x67, 0x65, 0x5f, 0x63, 0x6c, 0x61, 0x73, 0x73, 0x5f, 0x6e, 0x61, 0x6d, 0x65, 0x18,
	0x06, 0x20, 0x01, 0x28, 0x09, 0x52, 0x10, 0x73, 0x74, 0x6f, 0x72, 0x61, 0x67, 0x65, 0x43, 0x6c,
	0x61, 0x73, 0x73, 0x4e, 0x61, 0x6d, 0x65, 0x12, 0x1f, 0x0a, 0x0b, 0x76, 0x6f, 0x6c, 0x75, 0x6d,
	0x65, 0x5f, 0x6e, 0x61, 0x6d, 0x65, 0x18, 0x07, 0x20, 0x01, 0x28, 0x09, 0x52, 0x0a, 0x76, 0x6f,
	0x6c, 0x75, 0x6d, 0x65, 0x4e, 0x61, 0x6d, 0x65, 0x12, 0x39, 0x0a, 0x0b, 0x61, 0x6e, 0x6e, 0x6f,
	0x74, 0x61, 0x74, 0x69, 0x6f, 0x6e, 0x73, 0x18, 0x08, 0x20, 0x01, 0x28, 0x0b, 0x32, 0x17, 0x2e,
	0x67, 0x6f, 0x6f, 0x67, 0x6c, 0x65, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x62, 0x75, 0x66, 0x2e,
	0x53, 0x74, 0x72, 0x75, 0x63, 0x74, 0x52, 0x0b, 0x61, 0x6e, 0x6e, 0x6f, 0x74, 0x61, 0x74, 0x69,
	0x6f, 0x6e, 0x73, 0x42, 0x06, 0x0a, 0x04, 0x6e, 0x61, 0x6d, 0x65, 0x22, 0xd7, 0x01, 0x0a, 0x09,
	0x44, 0x65, 0x6c, 0x65, 0x74, 0x65, 0x50, 0x76, 0x63, 0x12, 0x5d, 0x0a, 0x15, 0x74, 0x61, 0x73,
	0x6b, 0x5f, 0x6f, 0x75, 0x74, 0x70, 0x75, 0x74, 0x5f, 0x70, 0x61, 0x72, 0x61, 0x6d, 0x65, 0x74,
	0x65, 0x72, 0x18, 0x01, 0x20, 0x01, 0x28, 0x0b, 0x32, 0x27, 0x2e, 0x6b, 0x66, 0x70, 0x5f, 0x6b,
	0x75, 0x62, 0x65, 0x72, 0x6e, 0x65, 0x74, 0x65, 0x73, 0x2e, 0x54, 0x61, 0x73, 0x6b, 0x4f, 0x75,
	0x74, 0x70, 0x75, 0x74, 0x50, 0x61, 0x72, 0x61, 0x6d, 0x65, 0x74, 0x65, 0x72, 0x53, 0x70, 0x65,
	0x63, 0x48, 0x00, 0x52, 0x13, 0x74, 0x61, 0x73, 0x6b, 0x4f, 0x75, 0x74, 0x70, 0x75, 0x74, 0x50,
	0x61, 0x72, 0x61, 0x6d, 0x65, 0x74, 0x65, 0x72, 0x12, 0x1c, 0x0a, 0x08, 0x63, 0x6f, 0x6e, 0x73,
	0x74, 0x61, 0x6e, 0x74, 0x18, 0x02, 0x20, 0x01, 0x28, 0x09, 0x48, 0x00, 0x52, 0x08, 0x63, 0x6f,
	0x6e, 0x73, 0x74, 0x61, 0x6e, 0x74, 0x12, 0x3c, 0x0a, 0x19, 0x63, 0x6f, 0x6d, 0x70, 0x6f, 0x6e,
	0x65, 0x6e, 0x74, 0x5f, 0x69, 0x6e, 0x70, 0x75, 0x74, 0x5f, 0x70, 0x61, 0x72, 0x61, 0x6d, 0x65,
	0x74, 0x65, 0x72, 0x18, 0x03, 0x20, 0x01, 0x28, 0x09, 0x48, 0x00, 0x52, 0x17, 0x63, 0x6f, 0x6d,
	0x70, 0x6f, 0x6e, 0x65, 0x6e, 0x74, 0x49, 0x6e, 0x70, 0x75, 0x74, 0x50, 0x61, 0x72, 0x61, 0x6d,
	0x65, 0x74, 0x65, 0x72, 0x42, 0x0f, 0x0a, 0x0d, 0x70, 0x76, 0x63, 0x5f, 0x72, 0x65, 0x66, 0x65,
	0x72, 0x65, 0x6e, 0x63, 0x65, 0x22, 0x4c, 0x0a, 0x0c, 0x4e, 0x6f, 0x64, 0x65, 0x53, 0x65, 0x6c,
	0x65, 0x63, 0x74, 0x6f, 0x72, 0x12, 0x1b, 0x0a, 0x09, 0x6c, 0x61, 0x62, 0x65, 0x6c, 0x5f, 0x6b,
	0x65, 0x79, 0x18, 0x01, 0x20, 0x01, 0x28, 0x09, 0x52, 0x08, 0x6c, 0x61, 0x62, 0x65, 0x6c, 0x4b,
	0x65, 0x79, 0x12, 0x1f, 0x0a, 0x0b, 0x6c, 0x61, 0x62, 0x65, 0x6c, 0x5f, 0x76, 0x61, 0x6c, 0x75,
	0x65, 0x18, 0x02, 0x20, 0x01, 0x28, 0x09, 0x52, 0x0a, 0x6c, 0x61, 0x62, 0x65, 0x6c, 0x56, 0x61,
	0x6c, 0x75, 0x65, 0x42, 0x49, 0x5a, 0x47, 0x67, 0x69, 0x74, 0x68, 0x75, 0x62, 0x2e, 0x63, 0x6f,
	0x6d, 0x2f, 0x6b, 0x75, 0x62, 0x65, 0x66, 0x6c, 0x6f, 0x77, 0x2f, 0x70, 0x69, 0x70, 0x65, 0x6c,
	0x69, 0x6e, 0x65, 0x73, 0x2f, 0x6b, 0x75, 0x62, 0x65, 0x72, 0x6e, 0x65, 0x74, 0x65, 0x73, 0x5f,
	0x70, 0x6c, 0x61, 0x74, 0x66, 0x6f, 0x72, 0x6d, 0x2f, 0x67, 0x6f, 0x2f, 0x6b, 0x75, 0x62, 0x65,
	0x72, 0x6e, 0x65, 0x74, 0x65, 0x73, 0x70, 0x6c, 0x61, 0x74, 0x66, 0x6f, 0x72, 0x6d, 0x62, 0x06,
	0x70, 0x72, 0x6f, 0x74, 0x6f, 0x33,
}

var (
	file_kubernetes_executor_config_proto_rawDescOnce sync.Once
	file_kubernetes_executor_config_proto_rawDescData = file_kubernetes_executor_config_proto_rawDesc
)

func file_kubernetes_executor_config_proto_rawDescGZIP() []byte {
	file_kubernetes_executor_config_proto_rawDescOnce.Do(func() {
		file_kubernetes_executor_config_proto_rawDescData = protoimpl.X.CompressGZIP(file_kubernetes_executor_config_proto_rawDescData)
	})
	return file_kubernetes_executor_config_proto_rawDescData
}

var file_kubernetes_executor_config_proto_msgTypes = make([]protoimpl.MessageInfo, 9)
var file_kubernetes_executor_config_proto_goTypes = []interface{}{
	(*KubernetesExecutorConfig)(nil),      // 0: kfp_kubernetes.KubernetesExecutorConfig
	(*SecretAsVolume)(nil),                // 1: kfp_kubernetes.SecretAsVolume
	(*SecretAsEnv)(nil),                   // 2: kfp_kubernetes.SecretAsEnv
	(*TaskOutputParameterSpec)(nil),       // 3: kfp_kubernetes.TaskOutputParameterSpec
	(*PvcMount)(nil),                      // 4: kfp_kubernetes.PvcMount
	(*CreatePvc)(nil),                     // 5: kfp_kubernetes.CreatePvc
	(*DeletePvc)(nil),                     // 6: kfp_kubernetes.DeletePvc
	(*NodeSelector)(nil),                  // 7: kfp_kubernetes.NodeSelector
	(*SecretAsEnv_SecretKeyToEnvMap)(nil), // 8: kfp_kubernetes.SecretAsEnv.SecretKeyToEnvMap
	(*structpb.Struct)(nil),               // 9: google.protobuf.Struct
}
var file_kubernetes_executor_config_proto_depIdxs = []int32{
	1, // 0: kfp_kubernetes.KubernetesExecutorConfig.secret_as_volume:type_name -> kfp_kubernetes.SecretAsVolume
	2, // 1: kfp_kubernetes.KubernetesExecutorConfig.secret_as_env:type_name -> kfp_kubernetes.SecretAsEnv
	4, // 2: kfp_kubernetes.KubernetesExecutorConfig.pvc_mount:type_name -> kfp_kubernetes.PvcMount
	7, // 3: kfp_kubernetes.KubernetesExecutorConfig.node_selector:type_name -> kfp_kubernetes.NodeSelector
	8, // 4: kfp_kubernetes.SecretAsEnv.key_to_env:type_name -> kfp_kubernetes.SecretAsEnv.SecretKeyToEnvMap
	3, // 5: kfp_kubernetes.PvcMount.task_output_parameter:type_name -> kfp_kubernetes.TaskOutputParameterSpec
	9, // 6: kfp_kubernetes.CreatePvc.annotations:type_name -> google.protobuf.Struct
	3, // 7: kfp_kubernetes.DeletePvc.task_output_parameter:type_name -> kfp_kubernetes.TaskOutputParameterSpec
	8, // [8:8] is the sub-list for method output_type
	8, // [8:8] is the sub-list for method input_type
	8, // [8:8] is the sub-list for extension type_name
	8, // [8:8] is the sub-list for extension extendee
	0, // [0:8] is the sub-list for field type_name
}

func init() { file_kubernetes_executor_config_proto_init() }
func file_kubernetes_executor_config_proto_init() {
	if File_kubernetes_executor_config_proto != nil {
		return
	}
	if !protoimpl.UnsafeEnabled {
		file_kubernetes_executor_config_proto_msgTypes[0].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*KubernetesExecutorConfig); i {
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
		file_kubernetes_executor_config_proto_msgTypes[1].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*SecretAsVolume); i {
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
		file_kubernetes_executor_config_proto_msgTypes[2].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*SecretAsEnv); i {
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
		file_kubernetes_executor_config_proto_msgTypes[3].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*TaskOutputParameterSpec); i {
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
		file_kubernetes_executor_config_proto_msgTypes[4].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*PvcMount); i {
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
		file_kubernetes_executor_config_proto_msgTypes[5].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*CreatePvc); i {
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
		file_kubernetes_executor_config_proto_msgTypes[6].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*DeletePvc); i {
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
		file_kubernetes_executor_config_proto_msgTypes[7].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*NodeSelector); i {
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
		file_kubernetes_executor_config_proto_msgTypes[8].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*SecretAsEnv_SecretKeyToEnvMap); i {
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
	file_kubernetes_executor_config_proto_msgTypes[4].OneofWrappers = []interface{}{
		(*PvcMount_TaskOutputParameter)(nil),
		(*PvcMount_Constant)(nil),
		(*PvcMount_ComponentInputParameter)(nil),
	}
	file_kubernetes_executor_config_proto_msgTypes[5].OneofWrappers = []interface{}{
		(*CreatePvc_PvcName)(nil),
		(*CreatePvc_PvcNameSuffix)(nil),
	}
	file_kubernetes_executor_config_proto_msgTypes[6].OneofWrappers = []interface{}{
		(*DeletePvc_TaskOutputParameter)(nil),
		(*DeletePvc_Constant)(nil),
		(*DeletePvc_ComponentInputParameter)(nil),
	}
	type x struct{}
	out := protoimpl.TypeBuilder{
		File: protoimpl.DescBuilder{
			GoPackagePath: reflect.TypeOf(x{}).PkgPath(),
			RawDescriptor: file_kubernetes_executor_config_proto_rawDesc,
			NumEnums:      0,
			NumMessages:   9,
			NumExtensions: 0,
			NumServices:   0,
		},
		GoTypes:           file_kubernetes_executor_config_proto_goTypes,
		DependencyIndexes: file_kubernetes_executor_config_proto_depIdxs,
		MessageInfos:      file_kubernetes_executor_config_proto_msgTypes,
	}.Build()
	File_kubernetes_executor_config_proto = out.File
	file_kubernetes_executor_config_proto_rawDesc = nil
	file_kubernetes_executor_config_proto_goTypes = nil
	file_kubernetes_executor_config_proto_depIdxs = nil
}
