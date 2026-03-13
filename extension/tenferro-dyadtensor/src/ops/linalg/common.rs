macro_rules! dispatch_linalg_runtime {
    ($ty:ty, $capability:expr, $op:literal, |$ctx:ident| $body:expr) => {
        with_linalg_runtime::<$ty, _>($op, $capability, |$ctx| $body, |$ctx| $body, |$ctx| $body)
    };
}

pub(crate) use dispatch_linalg_runtime;

macro_rules! dispatch_cpu_composite_runtime {
    ($op:literal, |$ctx:ident| $body:expr) => {
        with_runtime(
            |$ctx| $body,
            |_ctx| Err(unsupported_runtime_capability($op, "cuda")),
            |_ctx| Err(unsupported_runtime_capability($op, "rocm")),
        )
    };
}

pub(crate) use dispatch_cpu_composite_runtime;

macro_rules! unary_linalg_builder {
    (
        $builder:ident,
        $ctor:ident,
        returns = $ret:ty,
        capability = $capability:expr,
        op = $op:literal,
        bounds = ($($bounds:tt)+),
        call = |$ctx:ident, $self:ident| $body:expr
    ) => {
        #[doc = concat!("Builder for `", stringify!($ctor), "`.")]
        #[doc = ""]
        #[doc = "# Examples"]
        #[doc = ""]
        #[doc = concat!("```ignore\nlet _builder = ", stringify!($ctor), "(/* ... */);\n```")]
        #[derive(Clone, Copy)]
        pub struct $builder<'a, T: LinalgScalar> {
            tensor: &'a Tensor<T>,
        }

        impl<'a, T> $builder<'a, T>
        where
            $($bounds)+
        {
            #[doc = concat!("Executes `", stringify!($ctor), "`.")]
            #[doc = ""]
            #[doc = "# Examples"]
            #[doc = ""]
            #[doc = "```ignore\nlet _out = builder.run();\n```"]
            pub fn run(self) -> Result<$ret> {
                dispatch_linalg_runtime!(T, $capability, $op, |$ctx| {
                    let $self = &self;
                    $body
                })
            }
        }

        #[doc = concat!("Creates a `", stringify!($ctor), "` builder.")]
        #[doc = ""]
        #[doc = "# Examples"]
        #[doc = ""]
        #[doc = concat!("```ignore\nlet _ = ", stringify!($ctor), "(/* ... */);\n```")]
        pub fn $ctor<'a, T: LinalgScalar>(tensor: &'a Tensor<T>) -> $builder<'a, T> {
            $builder { tensor }
        }
    };
}

pub(crate) use unary_linalg_builder;

macro_rules! binary_linalg_builder {
    (
        $builder:ident,
        $ctor:ident,
        returns = $ret:ty,
        capability = $capability:expr,
        op = $op:literal,
        bounds = ($($bounds:tt)+),
        call = |$ctx:ident, $self:ident| $body:expr
    ) => {
        #[doc = concat!("Builder for `", stringify!($ctor), "`.")]
        #[doc = ""]
        #[doc = "# Examples"]
        #[doc = ""]
        #[doc = concat!("```ignore\nlet _builder = ", stringify!($ctor), "(/* ... */);\n```")]
        #[derive(Clone, Copy)]
        pub struct $builder<'a, T: LinalgScalar> {
            a: &'a Tensor<T>,
            b: &'a Tensor<T>,
        }

        impl<'a, T> $builder<'a, T>
        where
            $($bounds)+
        {
            #[doc = concat!("Executes `", stringify!($ctor), "`.")]
            #[doc = ""]
            #[doc = "# Examples"]
            #[doc = ""]
            #[doc = "```ignore\nlet _out = builder.run();\n```"]
            pub fn run(self) -> Result<$ret> {
                dispatch_linalg_runtime!(T, $capability, $op, |$ctx| {
                    let $self = &self;
                    $body
                })
            }
        }

        #[doc = concat!("Creates a `", stringify!($ctor), "` builder.")]
        #[doc = ""]
        #[doc = "# Examples"]
        #[doc = ""]
        #[doc = concat!("```ignore\nlet _ = ", stringify!($ctor), "(/* ... */);\n```")]
        pub fn $ctor<'a, T: LinalgScalar>(a: &'a Tensor<T>, b: &'a Tensor<T>) -> $builder<'a, T> {
            $builder { a, b }
        }
    };
}

pub(crate) use binary_linalg_builder;
