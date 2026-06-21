//! Minimal PJRT C API declarations used by the dynamic plugin loader and
//! single-device execution path.

#![allow(dead_code)]

use std::ffi::{c_char, c_void};

/// Opaque PJRT C API table.
#[repr(C)]
#[allow(non_camel_case_types)]
pub(crate) struct PJRT_Api {
    pub(crate) struct_size: usize,
    pub(crate) extension_start: *mut PJRT_Extension_Base,
    pub(crate) pjrt_api_version: PJRT_Api_Version,
    pub(crate) pjrt_error_destroy: PjrtErrorDestroy,
    pub(crate) pjrt_error_message: PjrtErrorMessage,
    pub(crate) pjrt_error_get_code: *const c_void,
    pub(crate) pjrt_plugin_initialize: *const c_void,
    pub(crate) pjrt_plugin_attributes: *const c_void,
    pub(crate) pjrt_event_destroy: PjrtEventDestroy,
    pub(crate) pjrt_event_is_ready: *const c_void,
    pub(crate) pjrt_event_error: PjrtEventError,
    pub(crate) pjrt_event_await: PjrtEventAwait,
    pub(crate) pjrt_event_on_ready: *const c_void,
    pub(crate) pjrt_client_create: PjrtClientCreate,
    pub(crate) pjrt_client_destroy: PjrtClientDestroy,
    pub(crate) pjrt_client_platform_name: *const c_void,
    pub(crate) pjrt_client_process_index: *const c_void,
    pub(crate) pjrt_client_platform_version: *const c_void,
    pub(crate) pjrt_client_devices: *const c_void,
    pub(crate) pjrt_client_addressable_devices: PjrtClientAddressableDevices,
    pub(crate) pjrt_client_lookup_device: *const c_void,
    pub(crate) pjrt_client_lookup_addressable_device: *const c_void,
    pub(crate) pjrt_client_addressable_memories: *const c_void,
    pub(crate) pjrt_client_compile: PjrtClientCompile,
    pub(crate) pjrt_client_default_device_assignment: *const c_void,
    pub(crate) pjrt_client_buffer_from_host_buffer: PjrtClientBufferFromHostBuffer,
    pub(crate) pjrt_device_description_id: *const c_void,
    pub(crate) pjrt_device_description_process_index: *const c_void,
    pub(crate) pjrt_device_description_attributes: *const c_void,
    pub(crate) pjrt_device_description_kind: *const c_void,
    pub(crate) pjrt_device_description_debug_string: *const c_void,
    pub(crate) pjrt_device_description_to_string: *const c_void,
    pub(crate) pjrt_device_get_description: *const c_void,
    pub(crate) pjrt_device_is_addressable: *const c_void,
    pub(crate) pjrt_device_local_hardware_id: *const c_void,
    pub(crate) pjrt_device_addressable_memories: *const c_void,
    pub(crate) pjrt_device_default_memory: *const c_void,
    pub(crate) pjrt_device_memory_stats: *const c_void,
    pub(crate) pjrt_memory_id: *const c_void,
    pub(crate) pjrt_memory_kind: *const c_void,
    pub(crate) pjrt_memory_debug_string: *const c_void,
    pub(crate) pjrt_memory_to_string: *const c_void,
    pub(crate) pjrt_memory_addressable_by_devices: *const c_void,
    pub(crate) pjrt_executable_destroy: PjrtExecutableDestroy,
    pub(crate) pjrt_executable_name: *const c_void,
    pub(crate) pjrt_executable_num_replicas: *const c_void,
    pub(crate) pjrt_executable_num_partitions: *const c_void,
    pub(crate) pjrt_executable_num_outputs: PjrtExecutableNumOutputs,
    pub(crate) pjrt_executable_size_of_generated_code_in_bytes: *const c_void,
    pub(crate) pjrt_executable_get_cost_analysis: *const c_void,
    pub(crate) pjrt_executable_output_memory_kinds: *const c_void,
    pub(crate) pjrt_executable_optimized_program: *const c_void,
    pub(crate) pjrt_executable_serialize: *const c_void,
    pub(crate) pjrt_loaded_executable_destroy: PjrtLoadedExecutableDestroy,
    pub(crate) pjrt_loaded_executable_get_executable: PjrtLoadedExecutableGetExecutable,
    pub(crate) pjrt_loaded_executable_addressable_devices: *const c_void,
    pub(crate) pjrt_loaded_executable_delete: *const c_void,
    pub(crate) pjrt_loaded_executable_is_deleted: *const c_void,
    pub(crate) pjrt_loaded_executable_execute: PjrtLoadedExecutableExecute,
    pub(crate) pjrt_executable_deserialize_and_load: *const c_void,
    pub(crate) pjrt_loaded_executable_fingerprint: *const c_void,
    pub(crate) pjrt_buffer_destroy: PjrtBufferDestroy,
    pub(crate) pjrt_buffer_element_type: *const c_void,
    pub(crate) pjrt_buffer_dimensions: *const c_void,
    pub(crate) pjrt_buffer_unpadded_dimensions: *const c_void,
    pub(crate) pjrt_buffer_dynamic_dimension_indices: *const c_void,
    pub(crate) pjrt_buffer_get_memory_layout: *const c_void,
    pub(crate) pjrt_buffer_on_device_size_in_bytes: *const c_void,
    pub(crate) pjrt_buffer_device: *const c_void,
    pub(crate) pjrt_buffer_memory: *const c_void,
    pub(crate) pjrt_buffer_delete: *const c_void,
    pub(crate) pjrt_buffer_is_deleted: *const c_void,
    pub(crate) pjrt_buffer_copy_to_device: *const c_void,
    pub(crate) pjrt_buffer_to_host_buffer: PjrtBufferToHostBuffer,
}

/// Plugin entry point exported by OpenXLA PJRT plugins.
#[allow(non_camel_case_types)]
pub(crate) type GetPjrtApiFn = unsafe extern "C" fn() -> *const PJRT_Api;

#[repr(C)]
#[allow(non_camel_case_types)]
pub(crate) struct PJRT_Extension_Base {
    _private: [u8; 0],
}

#[repr(C)]
#[allow(non_camel_case_types)]
#[derive(Clone, Copy)]
pub(crate) struct PJRT_Api_Version {
    pub(crate) struct_size: usize,
    pub(crate) extension_start: *mut PJRT_Extension_Base,
    pub(crate) major_version: i32,
    pub(crate) minor_version: i32,
}

#[repr(C)]
#[allow(non_camel_case_types)]
pub(crate) struct PJRT_Error {
    _private: [u8; 0],
}

#[repr(C)]
#[allow(non_camel_case_types)]
pub(crate) struct PJRT_Event {
    _private: [u8; 0],
}

#[repr(C)]
#[allow(non_camel_case_types)]
pub(crate) struct PJRT_Client {
    _private: [u8; 0],
}

#[repr(C)]
#[allow(non_camel_case_types)]
pub(crate) struct PJRT_Device {
    _private: [u8; 0],
}

#[repr(C)]
#[allow(non_camel_case_types)]
pub(crate) struct PJRT_Memory {
    _private: [u8; 0],
}

#[repr(C)]
#[allow(non_camel_case_types)]
pub(crate) struct PJRT_LoadedExecutable {
    _private: [u8; 0],
}

#[repr(C)]
#[allow(non_camel_case_types)]
pub(crate) struct PJRT_Executable {
    _private: [u8; 0],
}

#[repr(C)]
#[allow(non_camel_case_types)]
pub(crate) struct PJRT_Buffer {
    _private: [u8; 0],
}

#[repr(C)]
#[allow(non_camel_case_types)]
#[derive(Clone, Copy)]
pub(crate) enum PJRT_Buffer_MemoryLayout_Type {
    Tiled = 0,
    Strides = 1,
}

#[repr(C)]
#[allow(non_camel_case_types)]
#[derive(Clone, Copy)]
pub(crate) struct PJRT_Buffer_MemoryLayout_Tiled {
    pub(crate) struct_size: usize,
    pub(crate) extension_start: *mut PJRT_Extension_Base,
    pub(crate) minor_to_major: *const i64,
    pub(crate) minor_to_major_size: usize,
    pub(crate) tile_dims: *const i64,
    pub(crate) tile_dim_sizes: *const usize,
    pub(crate) num_tiles: usize,
}

#[repr(C)]
#[allow(non_camel_case_types)]
#[derive(Clone, Copy)]
pub(crate) struct PJRT_Buffer_MemoryLayout_Strides {
    pub(crate) struct_size: usize,
    pub(crate) extension_start: *mut PJRT_Extension_Base,
    pub(crate) byte_strides: *const i64,
    pub(crate) num_byte_strides: usize,
}

#[repr(C)]
#[allow(non_camel_case_types)]
#[derive(Clone, Copy)]
pub(crate) union PJRT_Buffer_MemoryLayout_Union {
    pub(crate) tiled: PJRT_Buffer_MemoryLayout_Tiled,
    pub(crate) strides: PJRT_Buffer_MemoryLayout_Strides,
}

#[repr(C)]
#[allow(non_camel_case_types)]
#[derive(Clone, Copy)]
pub(crate) struct PJRT_Buffer_MemoryLayout {
    pub(crate) struct_size: usize,
    pub(crate) extension_start: *mut PJRT_Extension_Base,
    pub(crate) layout: PJRT_Buffer_MemoryLayout_Union,
    pub(crate) type_: PJRT_Buffer_MemoryLayout_Type,
}

#[repr(C)]
#[allow(non_camel_case_types)]
pub(crate) struct PJRT_ExecuteContext {
    _private: [u8; 0],
}

#[repr(C)]
#[allow(non_camel_case_types)]
pub(crate) struct PJRT_MultiSlice_Config {
    _private: [u8; 0],
}

#[repr(C)]
#[allow(non_camel_case_types)]
pub(crate) struct PJRT_Error_Destroy_Args {
    pub(crate) struct_size: usize,
    pub(crate) extension_start: *mut PJRT_Extension_Base,
    pub(crate) error: *mut PJRT_Error,
}

#[repr(C)]
#[allow(non_camel_case_types)]
pub(crate) struct PJRT_Error_Message_Args {
    pub(crate) struct_size: usize,
    pub(crate) extension_start: *mut PJRT_Extension_Base,
    pub(crate) error: *const PJRT_Error,
    pub(crate) message: *const c_char,
    pub(crate) message_size: usize,
}

#[repr(C)]
#[allow(non_camel_case_types)]
pub(crate) struct PJRT_Event_Destroy_Args {
    pub(crate) struct_size: usize,
    pub(crate) extension_start: *mut PJRT_Extension_Base,
    pub(crate) event: *mut PJRT_Event,
}

#[repr(C)]
#[allow(non_camel_case_types)]
pub(crate) struct PJRT_Event_Error_Args {
    pub(crate) struct_size: usize,
    pub(crate) extension_start: *mut PJRT_Extension_Base,
    pub(crate) event: *mut PJRT_Event,
}

#[repr(C)]
#[allow(non_camel_case_types)]
pub(crate) struct PJRT_Event_Await_Args {
    pub(crate) struct_size: usize,
    pub(crate) extension_start: *mut PJRT_Extension_Base,
    pub(crate) event: *mut PJRT_Event,
}

#[repr(C)]
#[allow(non_camel_case_types)]
pub(crate) struct PJRT_Client_Create_Args {
    pub(crate) struct_size: usize,
    pub(crate) extension_start: *mut PJRT_Extension_Base,
    pub(crate) create_options: *const c_void,
    pub(crate) num_options: usize,
    pub(crate) kv_get_callback: *const c_void,
    pub(crate) kv_get_user_arg: *mut c_void,
    pub(crate) kv_put_callback: *const c_void,
    pub(crate) kv_put_user_arg: *mut c_void,
    pub(crate) client: *mut PJRT_Client,
    pub(crate) kv_try_get_callback: *const c_void,
    pub(crate) kv_try_get_user_arg: *mut c_void,
}

#[repr(C)]
#[allow(non_camel_case_types)]
pub(crate) struct PJRT_Client_Destroy_Args {
    pub(crate) struct_size: usize,
    pub(crate) extension_start: *mut PJRT_Extension_Base,
    pub(crate) client: *mut PJRT_Client,
}

#[repr(C)]
#[allow(non_camel_case_types)]
pub(crate) struct PJRT_Client_AddressableDevices_Args {
    pub(crate) struct_size: usize,
    pub(crate) extension_start: *mut PJRT_Extension_Base,
    pub(crate) client: *mut PJRT_Client,
    pub(crate) addressable_devices: *const *mut PJRT_Device,
    pub(crate) num_addressable_devices: usize,
}

#[repr(C)]
#[allow(non_camel_case_types)]
pub(crate) struct PJRT_Program {
    pub(crate) struct_size: usize,
    pub(crate) extension_start: *mut PJRT_Extension_Base,
    pub(crate) code: *mut c_char,
    pub(crate) code_size: usize,
    pub(crate) format: *const c_char,
    pub(crate) format_size: usize,
}

#[repr(C)]
#[allow(non_camel_case_types)]
pub(crate) struct PJRT_Client_Compile_Args {
    pub(crate) struct_size: usize,
    pub(crate) extension_start: *mut PJRT_Extension_Base,
    pub(crate) client: *mut PJRT_Client,
    pub(crate) program: *const PJRT_Program,
    pub(crate) compile_options: *const c_char,
    pub(crate) compile_options_size: usize,
    pub(crate) executable: *mut PJRT_LoadedExecutable,
}

#[repr(C)]
#[allow(non_camel_case_types)]
#[derive(Clone, Copy)]
pub(crate) enum PJRT_Buffer_Type {
    Invalid = 0,
    Pred = 1,
    S8 = 2,
    S16 = 3,
    S32 = 4,
    S64 = 5,
    U8 = 6,
    U16 = 7,
    U32 = 8,
    U64 = 9,
    F16 = 10,
    F32 = 11,
    F64 = 12,
}

#[repr(C)]
#[allow(non_camel_case_types)]
#[derive(Clone, Copy)]
pub(crate) enum PJRT_HostBufferSemantics {
    ImmutableOnlyDuringCall = 0,
    ImmutableUntilTransferCompletes = 1,
    ImmutableZeroCopy = 2,
    MutableZeroCopy = 3,
}

#[repr(C)]
#[allow(non_camel_case_types)]
pub(crate) struct PJRT_Client_BufferFromHostBuffer_Args {
    pub(crate) struct_size: usize,
    pub(crate) extension_start: *mut PJRT_Extension_Base,
    pub(crate) client: *mut PJRT_Client,
    pub(crate) data: *const c_void,
    pub(crate) type_: PJRT_Buffer_Type,
    pub(crate) dims: *const i64,
    pub(crate) num_dims: usize,
    pub(crate) byte_strides: *const i64,
    pub(crate) num_byte_strides: usize,
    pub(crate) host_buffer_semantics: PJRT_HostBufferSemantics,
    pub(crate) device: *mut PJRT_Device,
    pub(crate) memory: *mut PJRT_Memory,
    pub(crate) device_layout: *mut PJRT_Buffer_MemoryLayout,
    pub(crate) done_with_host_buffer: *mut PJRT_Event,
    pub(crate) buffer: *mut PJRT_Buffer,
}

#[repr(C)]
#[allow(non_camel_case_types)]
pub(crate) struct PJRT_LoadedExecutable_Destroy_Args {
    pub(crate) struct_size: usize,
    pub(crate) extension_start: *mut PJRT_Extension_Base,
    pub(crate) executable: *mut PJRT_LoadedExecutable,
}

#[repr(C)]
#[allow(non_camel_case_types)]
pub(crate) struct PJRT_Executable_Destroy_Args {
    pub(crate) struct_size: usize,
    pub(crate) extension_start: *mut PJRT_Extension_Base,
    pub(crate) executable: *mut PJRT_Executable,
}

#[repr(C)]
#[allow(non_camel_case_types)]
pub(crate) struct PJRT_LoadedExecutable_GetExecutable_Args {
    pub(crate) struct_size: usize,
    pub(crate) extension_start: *mut PJRT_Extension_Base,
    pub(crate) loaded_executable: *mut PJRT_LoadedExecutable,
    pub(crate) executable: *mut PJRT_Executable,
}

#[repr(C)]
#[allow(non_camel_case_types)]
pub(crate) struct PJRT_Executable_NumOutputs_Args {
    pub(crate) struct_size: usize,
    pub(crate) extension_start: *mut PJRT_Extension_Base,
    pub(crate) executable: *mut PJRT_Executable,
    pub(crate) num_outputs: usize,
}

#[repr(C)]
#[allow(non_camel_case_types)]
pub(crate) struct PJRT_ExecuteOptions {
    pub(crate) struct_size: usize,
    pub(crate) extension_start: *mut PJRT_Extension_Base,
    pub(crate) send_callbacks: *mut *mut c_void,
    pub(crate) recv_callbacks: *mut *mut c_void,
    pub(crate) num_send_ops: usize,
    pub(crate) num_recv_ops: usize,
    pub(crate) launch_id: i32,
    pub(crate) non_donatable_input_indices: *const i64,
    pub(crate) num_non_donatable_input_indices: usize,
    pub(crate) context: *mut PJRT_ExecuteContext,
    pub(crate) call_location: *const c_char,
    pub(crate) num_tasks: usize,
    pub(crate) task_ids: *mut i32,
    pub(crate) incarnation_ids: *mut i64,
    pub(crate) multi_slice_config: *mut PJRT_MultiSlice_Config,
}

#[repr(C)]
#[allow(non_camel_case_types)]
pub(crate) struct PJRT_LoadedExecutable_Execute_Args {
    pub(crate) struct_size: usize,
    pub(crate) extension_start: *mut PJRT_Extension_Base,
    pub(crate) executable: *mut PJRT_LoadedExecutable,
    pub(crate) options: *mut PJRT_ExecuteOptions,
    pub(crate) argument_lists: *const *const *mut PJRT_Buffer,
    pub(crate) num_devices: usize,
    pub(crate) num_args: usize,
    pub(crate) output_lists: *const *mut *mut PJRT_Buffer,
    pub(crate) device_complete_events: *mut *mut PJRT_Event,
    pub(crate) execute_device: *mut PJRT_Device,
}

#[repr(C)]
#[allow(non_camel_case_types)]
pub(crate) struct PJRT_Buffer_Destroy_Args {
    pub(crate) struct_size: usize,
    pub(crate) extension_start: *mut PJRT_Extension_Base,
    pub(crate) buffer: *mut PJRT_Buffer,
}

#[repr(C)]
#[allow(non_camel_case_types)]
pub(crate) struct PJRT_Buffer_ToHostBuffer_Args {
    pub(crate) struct_size: usize,
    pub(crate) extension_start: *mut PJRT_Extension_Base,
    pub(crate) src: *mut PJRT_Buffer,
    pub(crate) host_layout: *mut PJRT_Buffer_MemoryLayout,
    pub(crate) dst: *mut c_void,
    pub(crate) dst_size: usize,
    pub(crate) event: *mut PJRT_Event,
}

pub(crate) type PjrtErrorDestroy = unsafe extern "C" fn(*mut PJRT_Error_Destroy_Args);
pub(crate) type PjrtErrorMessage = unsafe extern "C" fn(*mut PJRT_Error_Message_Args);
pub(crate) type PjrtEventDestroy =
    unsafe extern "C" fn(*mut PJRT_Event_Destroy_Args) -> *mut PJRT_Error;
pub(crate) type PjrtEventError =
    unsafe extern "C" fn(*mut PJRT_Event_Error_Args) -> *mut PJRT_Error;
pub(crate) type PjrtEventAwait =
    unsafe extern "C" fn(*mut PJRT_Event_Await_Args) -> *mut PJRT_Error;
pub(crate) type PjrtClientCreate =
    unsafe extern "C" fn(*mut PJRT_Client_Create_Args) -> *mut PJRT_Error;
pub(crate) type PjrtClientDestroy =
    unsafe extern "C" fn(*mut PJRT_Client_Destroy_Args) -> *mut PJRT_Error;
pub(crate) type PjrtClientAddressableDevices =
    unsafe extern "C" fn(*mut PJRT_Client_AddressableDevices_Args) -> *mut PJRT_Error;
pub(crate) type PjrtClientCompile =
    unsafe extern "C" fn(*mut PJRT_Client_Compile_Args) -> *mut PJRT_Error;
pub(crate) type PjrtClientBufferFromHostBuffer =
    unsafe extern "C" fn(*mut PJRT_Client_BufferFromHostBuffer_Args) -> *mut PJRT_Error;
pub(crate) type PjrtLoadedExecutableDestroy =
    unsafe extern "C" fn(*mut PJRT_LoadedExecutable_Destroy_Args) -> *mut PJRT_Error;
pub(crate) type PjrtExecutableDestroy =
    unsafe extern "C" fn(*mut PJRT_Executable_Destroy_Args) -> *mut PJRT_Error;
pub(crate) type PjrtLoadedExecutableGetExecutable =
    unsafe extern "C" fn(*mut PJRT_LoadedExecutable_GetExecutable_Args) -> *mut PJRT_Error;
pub(crate) type PjrtLoadedExecutableExecute =
    unsafe extern "C" fn(*mut PJRT_LoadedExecutable_Execute_Args) -> *mut PJRT_Error;
pub(crate) type PjrtExecutableNumOutputs =
    unsafe extern "C" fn(*mut PJRT_Executable_NumOutputs_Args) -> *mut PJRT_Error;
pub(crate) type PjrtBufferDestroy =
    unsafe extern "C" fn(*mut PJRT_Buffer_Destroy_Args) -> *mut PJRT_Error;
pub(crate) type PjrtBufferToHostBuffer =
    unsafe extern "C" fn(*mut PJRT_Buffer_ToHostBuffer_Args) -> *mut PJRT_Error;
