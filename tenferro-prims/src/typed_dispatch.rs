macro_rules! cast_strided_view {
    ($view:expr, $from:ty, $to:ty) => {{
        unsafe {
            &*($view as *const strided_view::StridedView<$from>
                as *const strided_view::StridedView<$to>)
        }
    }};
}

pub(crate) use cast_strided_view;

macro_rules! cast_strided_view_mut {
    ($view:expr, $from:ty, $to:ty) => {{
        unsafe {
            &mut *($view as *mut strided_view::StridedViewMut<$from>
                as *mut strided_view::StridedViewMut<$to>)
        }
    }};
}

pub(crate) use cast_strided_view_mut;

macro_rules! cast_scalar_value {
    ($value:expr, $from:ty, $to:ty) => {{
        unsafe { *(&$value as *const $from as *const $to) }
    }};
}

pub(crate) use cast_scalar_value;

macro_rules! dispatch_standard_scalar_type {
    ($generic:ty, $concrete:ident, $body:block) => {{
        let tid = std::any::TypeId::of::<$generic>();
        if tid == std::any::TypeId::of::<f64>() {
            type $concrete = f64;
            $body
        }
        if tid == std::any::TypeId::of::<f32>() {
            type $concrete = f32;
            $body
        }
        if tid == std::any::TypeId::of::<num_complex::Complex64>() {
            type $concrete = num_complex::Complex64;
            $body
        }
        if tid == std::any::TypeId::of::<num_complex::Complex32>() {
            type $concrete = num_complex::Complex32;
            $body
        }
    }};
}

pub(crate) use dispatch_standard_scalar_type;

macro_rules! dispatch_real_scalar_type {
    ($generic:ty, $concrete:ident, $body:block) => {{
        let tid = std::any::TypeId::of::<$generic>();
        if tid == std::any::TypeId::of::<f64>() {
            type $concrete = f64;
            $body
        }
        if tid == std::any::TypeId::of::<f32>() {
            type $concrete = f32;
            $body
        }
    }};
}

pub(crate) use dispatch_real_scalar_type;

macro_rules! dispatch_complex_scalar_type {
    ($generic:ty, $concrete:ident, $body:block) => {{
        let tid = std::any::TypeId::of::<$generic>();
        if tid == std::any::TypeId::of::<num_complex::Complex64>() {
            type $concrete = num_complex::Complex64;
            $body
        }
        if tid == std::any::TypeId::of::<num_complex::Complex32>() {
            type $concrete = num_complex::Complex32;
            $body
        }
    }};
}

pub(crate) use dispatch_complex_scalar_type;
