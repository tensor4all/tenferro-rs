macro_rules! dispatch_type_id {
    ($tid:expr, $concrete:ident, [$($ty:ty),+ $(,)?], $body:block) => {{
        $(
            if $tid == std::any::TypeId::of::<$ty>() {
                type $concrete = $ty;
                $body
            }
        ) else+
    }};
}

pub(crate) use dispatch_type_id;

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
        $crate::infra::typed_dispatch::dispatch_type_id!(
            tid,
            $concrete,
            [f64, f32, num_complex::Complex64, num_complex::Complex32],
            $body
        )
    }};
}

pub(crate) use dispatch_standard_scalar_type;

macro_rules! dispatch_real_scalar_type {
    ($generic:ty, $concrete:ident, $body:block) => {{
        let tid = std::any::TypeId::of::<$generic>();
        $crate::infra::typed_dispatch::dispatch_type_id!(tid, $concrete, [f64, f32], $body)
    }};
}

pub(crate) use dispatch_real_scalar_type;

macro_rules! dispatch_complex_scalar_type {
    ($generic:ty, $concrete:ident, $body:block) => {{
        let tid = std::any::TypeId::of::<$generic>();
        $crate::infra::typed_dispatch::dispatch_type_id!(
            tid,
            $concrete,
            [num_complex::Complex64, num_complex::Complex32],
            $body
        )
    }};
}

pub(crate) use dispatch_complex_scalar_type;
