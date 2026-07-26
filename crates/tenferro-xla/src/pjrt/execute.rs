use std::ffi::{c_char, c_void};
use std::mem;
use std::ptr;
use std::slice;

use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::shape_extent::ShapeExtent;
use tenferro_runtime::program::{ProgramValue, SemanticProgram};
use tenferro_tensor::{DType, Tensor, TypedTensor};

use crate::layout::col_major_byte_strides;
use crate::{lower_to_stablehlo, Error, Result};

use super::plugin::PjrtPlugin;
use super::sys::*;

#[derive(Clone, Debug)]
struct TensorSpec {
    dtype: DType,
    shape: Vec<usize>,
}

pub(crate) fn run_many_with_inputs(
    plugin: &PjrtPlugin,
    program: &SemanticProgram,
    inputs: &[&Tensor],
) -> Result<Vec<Tensor>> {
    validate_inputs(program, inputs)?;
    let output_specs = output_specs(program)?;
    let module = lower_to_stablehlo(program)?;
    let api = plugin.api();
    if api.is_null() {
        return Err(Error::PjrtCall {
            call: "GetPjrtApi",
            message: "plugin returned null API table".to_string(),
        });
    }
    let api = unsafe { &*api };

    let mut client = PjrtClient::create(api)?;
    let device = client.first_addressable_device()?;
    let mut executable = client.compile(module.as_str(), &output_specs)?;
    let num_outputs = executable.num_outputs()?;
    if num_outputs != output_specs.len() {
        return Err(Error::InvalidProgram {
            message: format!(
                "PJRT executable reports {num_outputs} outputs but graph has {}",
                output_specs.len()
            ),
        });
    }

    let mut input_buffers = inputs
        .iter()
        .map(|input| client.buffer_from_host_tensor(input, device))
        .collect::<Result<Vec<_>>>()?;
    let mut output_buffers = executable.execute(device, &mut input_buffers, num_outputs)?;
    output_buffers
        .iter_mut()
        .zip(output_specs.iter())
        .map(|(buffer, spec)| buffer.download_tensor(spec))
        .collect()
}

fn validate_inputs(program: &SemanticProgram, inputs: &[&Tensor]) -> Result<()> {
    if inputs.len() != program.inputs().len() {
        return Err(Error::InvalidProgram {
            message: format!(
                "PJRT execution expected {} inputs, got {}",
                program.inputs().len(),
                inputs.len()
            ),
        });
    }
    for (index, (&value, input)) in program.inputs().iter().zip(inputs).enumerate() {
        let spec = semantic_tensor_spec(program, value, "ProgramInput", index, "PJRT input")?;
        if spec.dtype != input.dtype() {
            return Err(Error::InvalidProgram {
                message: format!(
                    "PJRT input {index} expected dtype {:?}, got {:?}",
                    spec.dtype,
                    input.dtype()
                ),
            });
        }
        if spec.shape != input.shape() {
            return Err(Error::InvalidProgram {
                message: format!(
                    "PJRT input {index} expected shape {:?}, got {:?}",
                    spec.shape,
                    input.shape()
                ),
            });
        }
        validate_supported_dtype(input.dtype(), "PJRT input")?;
    }
    Ok(())
}

fn output_specs(program: &SemanticProgram) -> Result<Vec<TensorSpec>> {
    program
        .outputs()
        .iter()
        .enumerate()
        .map(|(index, &value)| {
            semantic_tensor_spec(program, value, "ProgramOutput", index, "PJRT output")
        })
        .collect()
}

fn semantic_tensor_spec(
    program: &SemanticProgram,
    value: ProgramValue,
    op: &'static str,
    output_index: usize,
    context: &'static str,
) -> Result<TensorSpec> {
    let metadata = program
        .value_metadata(value)
        .map_err(|source| Error::InvalidProgram {
            message: format!("{context} semantic metadata is unavailable: {source}"),
        })?;
    validate_supported_dtype(metadata.dtype(), context)?;
    let shape = metadata
        .shape()
        .iter()
        .enumerate()
        .map(|(axis, extent)| match extent {
            ShapeExtent::Exact(DimExpr::Const(value)) => Ok(*value),
            ShapeExtent::Exact(_) => Err(Error::NonStaticShape {
                op,
                output_index,
                axis,
                kind: "symbolic",
            }),
            ShapeExtent::UpperBound(_) => Err(Error::NonStaticShape {
                op,
                output_index,
                axis,
                kind: "an upper bound",
            }),
            ShapeExtent::Unknown => Err(Error::NonStaticShape {
                op,
                output_index,
                axis,
                kind: "unknown",
            }),
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(TensorSpec {
        dtype: metadata.dtype(),
        shape,
    })
}

fn validate_supported_dtype(dtype: DType, context: &'static str) -> Result<()> {
    match dtype {
        DType::F32 | DType::F64 => Ok(()),
        other => Err(Error::UnsupportedDType {
            dtype: other,
            context,
        }),
    }
}

fn dims_i64(shape: &[usize]) -> Result<Vec<i64>> {
    shape
        .iter()
        .map(|&dim| {
            i64::try_from(dim).map_err(|_| Error::InvalidProgram {
                message: format!("shape dimension {dim} exceeds i64 for PJRT"),
            })
        })
        .collect()
}

fn element_count(shape: &[usize]) -> Result<usize> {
    shape.iter().try_fold(1_usize, |acc, &dim| {
        acc.checked_mul(dim).ok_or_else(|| Error::InvalidProgram {
            message: format!("shape {:?} element count overflows usize", shape),
        })
    })
}

fn tensor_slice<T: tenferro_tensor::TensorScalar>(tensor: &Tensor) -> Result<&[T]> {
    tensor.as_slice::<T>().map_err(Error::from)
}

fn pjrt_error_message(api: &PJRT_Api, error: *mut PJRT_Error) -> String {
    if error.is_null() {
        return String::new();
    }
    let mut args = PJRT_Error_Message_Args {
        struct_size: mem::size_of::<PJRT_Error_Message_Args>(),
        extension_start: ptr::null_mut(),
        error,
        message: ptr::null(),
        message_size: 0,
    };
    unsafe { (api.pjrt_error_message)(&mut args) };
    let message = if args.message.is_null() {
        "unknown PJRT error".to_string()
    } else {
        let bytes = unsafe { slice::from_raw_parts(args.message.cast::<u8>(), args.message_size) };
        String::from_utf8_lossy(bytes).into_owned()
    };
    let mut destroy = PJRT_Error_Destroy_Args {
        struct_size: mem::size_of::<PJRT_Error_Destroy_Args>(),
        extension_start: ptr::null_mut(),
        error,
    };
    unsafe { (api.pjrt_error_destroy)(&mut destroy) };
    message
}

fn check(api: &PJRT_Api, call: &'static str, error: *mut PJRT_Error) -> Result<()> {
    if error.is_null() {
        Ok(())
    } else {
        Err(Error::PjrtCall {
            call,
            message: pjrt_error_message(api, error),
        })
    }
}

struct PjrtClient<'a> {
    api: &'a PJRT_Api,
    ptr: *mut PJRT_Client,
}

impl<'a> PjrtClient<'a> {
    fn create(api: &'a PJRT_Api) -> Result<Self> {
        let mut args = PJRT_Client_Create_Args {
            struct_size: mem::size_of::<PJRT_Client_Create_Args>(),
            extension_start: ptr::null_mut(),
            create_options: ptr::null(),
            num_options: 0,
            kv_get_callback: ptr::null(),
            kv_get_user_arg: ptr::null_mut(),
            kv_put_callback: ptr::null(),
            kv_put_user_arg: ptr::null_mut(),
            client: ptr::null_mut(),
            kv_try_get_callback: ptr::null(),
            kv_try_get_user_arg: ptr::null_mut(),
        };
        let error = unsafe { (api.pjrt_client_create)(&mut args) };
        check(api, "PJRT_Client_Create", error)?;
        if args.client.is_null() {
            return Err(Error::PjrtCall {
                call: "PJRT_Client_Create",
                message: "returned null client".to_string(),
            });
        }
        Ok(Self {
            api,
            ptr: args.client,
        })
    }

    fn first_addressable_device(&mut self) -> Result<*mut PJRT_Device> {
        let mut args = PJRT_Client_AddressableDevices_Args {
            struct_size: mem::size_of::<PJRT_Client_AddressableDevices_Args>(),
            extension_start: ptr::null_mut(),
            client: self.ptr,
            addressable_devices: ptr::null(),
            num_addressable_devices: 0,
        };
        let error = unsafe { (self.api.pjrt_client_addressable_devices)(&mut args) };
        check(self.api, "PJRT_Client_AddressableDevices", error)?;
        if args.num_addressable_devices == 0 || args.addressable_devices.is_null() {
            return Err(Error::PjrtCall {
                call: "PJRT_Client_AddressableDevices",
                message: "client has no addressable devices".to_string(),
            });
        }
        let devices = unsafe {
            slice::from_raw_parts(args.addressable_devices, args.num_addressable_devices)
        };
        Ok(devices[0])
    }

    fn compile(
        &mut self,
        mlir: &str,
        output_specs: &[TensorSpec],
    ) -> Result<PjrtLoadedExecutable<'a>> {
        let mut code = mlir.as_bytes().to_vec();
        let format = b"mlir";
        let mut program = PJRT_Program {
            struct_size: mem::size_of::<PJRT_Program>(),
            extension_start: ptr::null_mut(),
            code: code.as_mut_ptr().cast::<c_char>(),
            code_size: code.len(),
            format: format.as_ptr().cast::<c_char>(),
            format_size: format.len(),
        };
        let compile_options = compile_options_proto(output_specs)?;
        let mut args = PJRT_Client_Compile_Args {
            struct_size: mem::size_of::<PJRT_Client_Compile_Args>(),
            extension_start: ptr::null_mut(),
            client: self.ptr,
            program: &mut program,
            compile_options: compile_options.as_ptr().cast::<c_char>(),
            compile_options_size: compile_options.len(),
            executable: ptr::null_mut(),
        };
        let error = unsafe { (self.api.pjrt_client_compile)(&mut args) };
        check(self.api, "PJRT_Client_Compile", error)?;
        if args.executable.is_null() {
            return Err(Error::PjrtCall {
                call: "PJRT_Client_Compile",
                message: "returned null executable".to_string(),
            });
        }
        Ok(PjrtLoadedExecutable {
            api: self.api,
            ptr: args.executable,
        })
    }

    fn buffer_from_host_tensor(
        &mut self,
        tensor: &Tensor,
        device: *mut PJRT_Device,
    ) -> Result<PjrtBuffer<'a>> {
        let shape = tensor.shape();
        let dims = dims_i64(shape)?;
        match tensor.dtype() {
            DType::F32 => {
                let data = tensor_slice::<f32>(tensor)?;
                let byte_strides = col_major_byte_strides::<f32>(shape)?;
                self.buffer_from_host_slice(
                    data.as_ptr().cast::<c_void>(),
                    PJRT_Buffer_Type::F32,
                    &dims,
                    &byte_strides,
                    device,
                )
            }
            DType::F64 => {
                let data = tensor_slice::<f64>(tensor)?;
                let byte_strides = col_major_byte_strides::<f64>(shape)?;
                self.buffer_from_host_slice(
                    data.as_ptr().cast::<c_void>(),
                    PJRT_Buffer_Type::F64,
                    &dims,
                    &byte_strides,
                    device,
                )
            }
            other => Err(Error::UnsupportedDType {
                dtype: other,
                context: "PJRT input",
            }),
        }
    }

    fn buffer_from_host_slice(
        &mut self,
        data: *const c_void,
        dtype: PJRT_Buffer_Type,
        dims: &[i64],
        byte_strides: &[i64],
        device: *mut PJRT_Device,
    ) -> Result<PjrtBuffer<'a>> {
        let mut args = PJRT_Client_BufferFromHostBuffer_Args {
            struct_size: mem::size_of::<PJRT_Client_BufferFromHostBuffer_Args>(),
            extension_start: ptr::null_mut(),
            client: self.ptr,
            data,
            type_: dtype,
            dims: dims.as_ptr(),
            num_dims: dims.len(),
            byte_strides: byte_strides.as_ptr(),
            num_byte_strides: byte_strides.len(),
            host_buffer_semantics: PJRT_HostBufferSemantics::ImmutableOnlyDuringCall,
            device,
            memory: ptr::null_mut(),
            device_layout: ptr::null_mut(),
            done_with_host_buffer: ptr::null_mut(),
            buffer: ptr::null_mut(),
        };
        let error = unsafe { (self.api.pjrt_client_buffer_from_host_buffer)(&mut args) };
        check(self.api, "PJRT_Client_BufferFromHostBuffer", error)?;
        if !args.done_with_host_buffer.is_null() {
            let mut event = PjrtEvent {
                api: self.api,
                ptr: args.done_with_host_buffer,
            };
            event.await_ready("PJRT_Client_BufferFromHostBuffer.done_with_host_buffer")?;
        }
        if args.buffer.is_null() {
            return Err(Error::PjrtCall {
                call: "PJRT_Client_BufferFromHostBuffer",
                message: "returned null buffer".to_string(),
            });
        }
        Ok(PjrtBuffer {
            api: self.api,
            ptr: args.buffer,
        })
    }
}

fn compile_options_proto(output_specs: &[TensorSpec]) -> Result<Vec<u8>> {
    let mut build_options = Vec::new();
    encode_message_field(
        &mut build_options,
        2,
        &result_layout_shape_proto(output_specs)?,
    );
    encode_varint_field(&mut build_options, 4, 1);
    encode_varint_field(&mut build_options, 5, 1);

    let mut options = Vec::new();
    encode_message_field(&mut options, 3, &build_options);
    Ok(options)
}

fn result_layout_shape_proto(output_specs: &[TensorSpec]) -> Result<Vec<u8>> {
    if output_specs.len() == 1 {
        return array_shape_proto(&output_specs[0]);
    }

    let mut shape = Vec::new();
    encode_varint_field(&mut shape, 2, 13);
    for spec in output_specs {
        encode_message_field(&mut shape, 4, &array_shape_proto(spec)?);
    }
    Ok(shape)
}

fn array_shape_proto(spec: &TensorSpec) -> Result<Vec<u8>> {
    let mut shape = Vec::new();
    encode_varint_field(&mut shape, 2, primitive_type_number(spec.dtype)?);
    for &dim in &spec.shape {
        let dim = u64::try_from(dim).map_err(|_| Error::InvalidProgram {
            message: format!("shape dimension {dim} exceeds u64 for XLA ShapeProto"),
        })?;
        encode_varint_field(&mut shape, 3, dim);
    }
    let mut layout = Vec::new();
    for axis in 0..spec.shape.len() {
        let axis = u64::try_from(axis).map_err(|_| Error::InvalidProgram {
            message: format!("axis {axis} exceeds u64 for XLA LayoutProto"),
        })?;
        encode_varint_field(&mut layout, 1, axis);
    }
    if !layout.is_empty() {
        encode_message_field(&mut shape, 5, &layout);
    }
    Ok(shape)
}

fn primitive_type_number(dtype: DType) -> Result<u64> {
    match dtype {
        DType::F32 => Ok(11),
        DType::F64 => Ok(12),
        other => Err(Error::UnsupportedDType {
            dtype: other,
            context: "PJRT result layout",
        }),
    }
}

fn encode_varint_field(out: &mut Vec<u8>, field: u64, value: u64) {
    encode_varint(out, field << 3);
    encode_varint(out, value);
}

fn encode_message_field(out: &mut Vec<u8>, field: u64, bytes: &[u8]) {
    encode_varint(out, (field << 3) | 2);
    encode_varint(out, bytes.len() as u64);
    out.extend_from_slice(bytes);
}

fn encode_varint(out: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        out.push((value as u8) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

impl Drop for PjrtClient<'_> {
    fn drop(&mut self) {
        if self.ptr.is_null() {
            return;
        }
        let mut args = PJRT_Client_Destroy_Args {
            struct_size: mem::size_of::<PJRT_Client_Destroy_Args>(),
            extension_start: ptr::null_mut(),
            client: self.ptr,
        };
        let _ = unsafe { (self.api.pjrt_client_destroy)(&mut args) };
    }
}

struct PjrtLoadedExecutable<'a> {
    api: &'a PJRT_Api,
    ptr: *mut PJRT_LoadedExecutable,
}

impl<'a> PjrtLoadedExecutable<'a> {
    fn num_outputs(&mut self) -> Result<usize> {
        let mut get = PJRT_LoadedExecutable_GetExecutable_Args {
            struct_size: mem::size_of::<PJRT_LoadedExecutable_GetExecutable_Args>(),
            extension_start: ptr::null_mut(),
            loaded_executable: self.ptr,
            executable: ptr::null_mut(),
        };
        let error = unsafe { (self.api.pjrt_loaded_executable_get_executable)(&mut get) };
        check(self.api, "PJRT_LoadedExecutable_GetExecutable", error)?;
        if get.executable.is_null() {
            return Err(Error::PjrtCall {
                call: "PJRT_LoadedExecutable_GetExecutable",
                message: "returned null executable".to_string(),
            });
        }
        let executable = PjrtExecutable {
            api: self.api,
            ptr: get.executable,
        };
        let mut args = PJRT_Executable_NumOutputs_Args {
            struct_size: mem::size_of::<PJRT_Executable_NumOutputs_Args>(),
            extension_start: ptr::null_mut(),
            executable: executable.ptr,
            num_outputs: 0,
        };
        let error = unsafe { (self.api.pjrt_executable_num_outputs)(&mut args) };
        check(self.api, "PJRT_Executable_NumOutputs", error)?;
        Ok(args.num_outputs)
    }

    fn execute(
        &mut self,
        device: *mut PJRT_Device,
        inputs: &mut [PjrtBuffer<'a>],
        num_outputs: usize,
    ) -> Result<Vec<PjrtBuffer<'a>>> {
        let input_ptrs = inputs
            .iter_mut()
            .map(|buffer| buffer.ptr)
            .collect::<Vec<_>>();
        let argument_lists = [input_ptrs.as_ptr()];
        let mut output_ptrs = vec![ptr::null_mut(); num_outputs];
        let output_lists = [output_ptrs.as_mut_ptr()];
        let mut complete_events = [ptr::null_mut()];
        let mut options = PJRT_ExecuteOptions {
            struct_size: mem::size_of::<PJRT_ExecuteOptions>(),
            extension_start: ptr::null_mut(),
            send_callbacks: ptr::null_mut(),
            recv_callbacks: ptr::null_mut(),
            num_send_ops: 0,
            num_recv_ops: 0,
            launch_id: 0,
            non_donatable_input_indices: ptr::null(),
            num_non_donatable_input_indices: 0,
            context: ptr::null_mut(),
            call_location: ptr::null(),
            num_tasks: 0,
            task_ids: ptr::null_mut(),
            incarnation_ids: ptr::null_mut(),
            multi_slice_config: ptr::null_mut(),
        };
        let mut args = PJRT_LoadedExecutable_Execute_Args {
            struct_size: mem::size_of::<PJRT_LoadedExecutable_Execute_Args>(),
            extension_start: ptr::null_mut(),
            executable: self.ptr,
            options: &mut options,
            argument_lists: argument_lists.as_ptr(),
            num_devices: 1,
            num_args: input_ptrs.len(),
            output_lists: output_lists.as_ptr(),
            device_complete_events: complete_events.as_mut_ptr(),
            execute_device: device,
        };
        let error = unsafe { (self.api.pjrt_loaded_executable_execute)(&mut args) };
        check(self.api, "PJRT_LoadedExecutable_Execute", error)?;
        if !complete_events[0].is_null() {
            let mut event = PjrtEvent {
                api: self.api,
                ptr: complete_events[0],
            };
            event.await_ready("PJRT_LoadedExecutable_Execute.device_complete")?;
        }
        output_ptrs
            .into_iter()
            .enumerate()
            .map(|(index, ptr)| {
                if ptr.is_null() {
                    Err(Error::PjrtCall {
                        call: "PJRT_LoadedExecutable_Execute",
                        message: format!("output buffer {index} is null"),
                    })
                } else {
                    Ok(PjrtBuffer { api: self.api, ptr })
                }
            })
            .collect()
    }
}

impl Drop for PjrtLoadedExecutable<'_> {
    fn drop(&mut self) {
        if self.ptr.is_null() {
            return;
        }
        let mut args = PJRT_LoadedExecutable_Destroy_Args {
            struct_size: mem::size_of::<PJRT_LoadedExecutable_Destroy_Args>(),
            extension_start: ptr::null_mut(),
            executable: self.ptr,
        };
        let _ = unsafe { (self.api.pjrt_loaded_executable_destroy)(&mut args) };
    }
}

struct PjrtExecutable<'a> {
    api: &'a PJRT_Api,
    ptr: *mut PJRT_Executable,
}

impl Drop for PjrtExecutable<'_> {
    fn drop(&mut self) {
        if self.ptr.is_null() {
            return;
        }
        let mut args = PJRT_Executable_Destroy_Args {
            struct_size: mem::size_of::<PJRT_Executable_Destroy_Args>(),
            extension_start: ptr::null_mut(),
            executable: self.ptr,
        };
        let _ = unsafe { (self.api.pjrt_executable_destroy)(&mut args) };
    }
}

struct PjrtBuffer<'a> {
    api: &'a PJRT_Api,
    ptr: *mut PJRT_Buffer,
}

impl PjrtBuffer<'_> {
    fn download_tensor(&mut self, spec: &TensorSpec) -> Result<Tensor> {
        match spec.dtype {
            DType::F32 => {
                let col_major = self.download_host_vec::<f32>(&spec.shape)?;
                let tensor = TypedTensor::from_vec_col_major(spec.shape.clone(), col_major)
                    .map_err(Error::from)?;
                Ok(Tensor::F32(tensor))
            }
            DType::F64 => {
                let col_major = self.download_host_vec::<f64>(&spec.shape)?;
                let tensor = TypedTensor::from_vec_col_major(spec.shape.clone(), col_major)
                    .map_err(Error::from)?;
                Ok(Tensor::F64(tensor))
            }
            other => Err(Error::UnsupportedDType {
                dtype: other,
                context: "PJRT output",
            }),
        }
    }

    fn download_host_vec<T: Copy + Default>(&mut self, shape: &[usize]) -> Result<Vec<T>> {
        let len = element_count(shape)?;
        let byte_len =
            len.checked_mul(mem::size_of::<T>())
                .ok_or_else(|| Error::InvalidProgram {
                    message: format!("shape {:?} byte length overflows usize", shape),
                })?;
        let mut output = vec![T::default(); len];
        let minor_to_major = col_major_minor_to_major(shape)?;
        let tiled = PJRT_Buffer_MemoryLayout_Tiled {
            struct_size: mem::size_of::<PJRT_Buffer_MemoryLayout_Tiled>(),
            extension_start: ptr::null_mut(),
            minor_to_major: minor_to_major.as_ptr(),
            minor_to_major_size: minor_to_major.len(),
            tile_dims: ptr::null(),
            tile_dim_sizes: ptr::null(),
            num_tiles: 0,
        };
        let mut host_layout = PJRT_Buffer_MemoryLayout {
            struct_size: mem::size_of::<PJRT_Buffer_MemoryLayout>(),
            extension_start: ptr::null_mut(),
            layout: PJRT_Buffer_MemoryLayout_Union { tiled },
            type_: PJRT_Buffer_MemoryLayout_Type::Tiled,
        };
        let mut args = PJRT_Buffer_ToHostBuffer_Args {
            struct_size: mem::size_of::<PJRT_Buffer_ToHostBuffer_Args>(),
            extension_start: ptr::null_mut(),
            src: self.ptr,
            host_layout: &mut host_layout,
            dst: output.as_mut_ptr().cast::<c_void>(),
            dst_size: byte_len,
            event: ptr::null_mut(),
        };
        let error = unsafe { (self.api.pjrt_buffer_to_host_buffer)(&mut args) };
        let mut event = PjrtEvent::from_raw(self.api, args.event);
        check(self.api, "PJRT_Buffer_ToHostBuffer", error)?;
        event.await_ready_if_present("PJRT_Buffer_ToHostBuffer.event")?;
        Ok(output)
    }
}

fn col_major_minor_to_major(shape: &[usize]) -> Result<Vec<i64>> {
    (0..shape.len())
        .map(|axis| {
            i64::try_from(axis).map_err(|_| Error::InvalidProgram {
                message: format!("axis {axis} exceeds i64 for PJRT host layout"),
            })
        })
        .collect()
}

impl Drop for PjrtBuffer<'_> {
    fn drop(&mut self) {
        if self.ptr.is_null() {
            return;
        }
        let mut args = PJRT_Buffer_Destroy_Args {
            struct_size: mem::size_of::<PJRT_Buffer_Destroy_Args>(),
            extension_start: ptr::null_mut(),
            buffer: self.ptr,
        };
        let _ = unsafe { (self.api.pjrt_buffer_destroy)(&mut args) };
    }
}

struct PjrtEvent<'a> {
    api: &'a PJRT_Api,
    ptr: *mut PJRT_Event,
}

impl PjrtEvent<'_> {
    fn from_raw(api: &PJRT_Api, ptr: *mut PJRT_Event) -> PjrtEvent<'_> {
        PjrtEvent { api, ptr }
    }

    fn await_ready_if_present(&mut self, call: &'static str) -> Result<()> {
        if self.ptr.is_null() {
            return Ok(());
        }
        self.await_ready(call)
    }

    fn await_ready(&mut self, call: &'static str) -> Result<()> {
        let mut args = PJRT_Event_Await_Args {
            struct_size: mem::size_of::<PJRT_Event_Await_Args>(),
            extension_start: ptr::null_mut(),
            event: self.ptr,
        };
        let error = unsafe { (self.api.pjrt_event_await)(&mut args) };
        check(self.api, call, error)
    }
}

impl Drop for PjrtEvent<'_> {
    fn drop(&mut self) {
        if self.ptr.is_null() {
            return;
        }
        let mut args = PJRT_Event_Destroy_Args {
            struct_size: mem::size_of::<PJRT_Event_Destroy_Args>(),
            extension_start: ptr::null_mut(),
            event: self.ptr,
        };
        let _ = unsafe { (self.api.pjrt_event_destroy)(&mut args) };
    }
}

#[cfg(test)]
mod tests {
    use tenferro_tensor::DType;

    use super::{compile_options_proto, TensorSpec};

    #[test]
    fn compile_options_proto_sets_column_major_result_layout() {
        let bytes = compile_options_proto(&[TensorSpec {
            dtype: DType::F32,
            shape: vec![2, 3],
        }])
        .unwrap();

        assert_eq!(
            bytes,
            vec![
                0x1a, 0x12, 0x12, 0x0c, 0x10, 0x0b, 0x18, 0x02, 0x18, 0x03, 0x2a, 0x04, 0x08, 0x00,
                0x08, 0x01, 0x20, 0x01, 0x28, 0x01,
            ]
        );
    }
}
