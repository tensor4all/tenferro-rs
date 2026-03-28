macro_rules! define_unary_ad_builder {
    ($builder:ident, $ctor:ident, $doc_op:literal, generic, |$self_ident:ident| $run:block) => {
        #[doc = concat!("Builder for AD `", $doc_op, "` on tensors.")]
        #[doc = ""]
        #[doc = "# Examples"]
        #[doc = ""]
        #[doc = concat!("```ignore\nlet out = tenferro::", stringify!($ctor), "(&x).run()?;\n```")]
        pub struct $builder<'a, T: Scalar> {
            tensor: &'a AdTensor<T>,
        }

        impl<'a, T> $builder<'a, T>
        where
            T: GenericAdRuntimeValue,
        {
            #[doc = concat!("Executes AD `", $doc_op, "`.")]
            #[doc = ""]
            #[doc = "# Examples"]
            #[doc = ""]
            #[doc = "```ignore\nlet out = builder.run()?;\n```"]
            pub fn run(self) -> Result<AdTensor<T>> {
                let $self_ident = self;
                $run
            }
        }

        #[doc = concat!("Creates a builder for AD `", $doc_op, "`.")]
        #[doc = ""]
        #[doc = "# Examples"]
        #[doc = ""]
        #[doc = concat!("```ignore\nlet out = tenferro::", stringify!($ctor), "(&x).run()?;\n```")]
        pub fn $ctor<'a, T>(tensor: &'a AdTensor<T>) -> $builder<'a, T>
        where
            T: GenericAdRuntimeValue,
        {
            $builder { tensor }
        }
    };
    ($builder:ident, $ctor:ident, $doc_op:literal, real, |$self_ident:ident| $run:block) => {
        #[doc = concat!("Builder for AD `", $doc_op, "` on tensors.")]
        #[doc = ""]
        #[doc = "# Examples"]
        #[doc = ""]
        #[doc = concat!("```ignore\nlet out = tenferro::", stringify!($ctor), "(&x).run()?;\n```")]
        pub struct $builder<'a, T: Scalar> {
            tensor: &'a AdTensor<T>,
        }

        impl<'a, T> $builder<'a, T>
        where
            T: RealAdRuntimeValue,
        {
            #[doc = concat!("Executes AD `", $doc_op, "`.")]
            #[doc = ""]
            #[doc = "# Examples"]
            #[doc = ""]
            #[doc = "```ignore\nlet out = builder.run()?;\n```"]
            pub fn run(self) -> Result<AdTensor<T>> {
                let $self_ident = self;
                $run
            }
        }

        #[doc = concat!("Creates a builder for AD `", $doc_op, "`.")]
        #[doc = ""]
        #[doc = "# Examples"]
        #[doc = ""]
        #[doc = concat!("```ignore\nlet out = tenferro::", stringify!($ctor), "(&x).run()?;\n```")]
        pub fn $ctor<'a, T>(tensor: &'a AdTensor<T>) -> $builder<'a, T>
        where
            T: RealAdRuntimeValue,
        {
            $builder { tensor }
        }
    };
}

pub(super) use define_unary_ad_builder;

macro_rules! define_binary_ad_builder {
    ($builder:ident, $ctor:ident, $doc_op:literal, generic, |$self_ident:ident| $run:block) => {
        #[doc = concat!("Builder for AD `", $doc_op, "` on tensors.")]
        #[doc = ""]
        #[doc = "# Examples"]
        #[doc = ""]
        #[doc = concat!("```ignore\nlet out = tenferro::", stringify!($ctor), "(&a, &b).run()?;\n```")]
        pub struct $builder<'a, T: Scalar> {
            lhs: &'a AdTensor<T>,
            rhs: &'a AdTensor<T>,
        }

        impl<'a, T> $builder<'a, T>
        where
            T: GenericAdRuntimeValue,
        {
            #[doc = concat!("Executes AD `", $doc_op, "`.")]
            #[doc = ""]
            #[doc = "# Examples"]
            #[doc = ""]
            #[doc = "```ignore\nlet out = builder.run()?;\n```"]
            pub fn run(self) -> Result<AdTensor<T>> {
                let $self_ident = self;
                $run
            }
        }

        #[doc = concat!("Creates a builder for AD `", $doc_op, "`.")]
        #[doc = ""]
        #[doc = "# Examples"]
        #[doc = ""]
        #[doc = concat!("```ignore\nlet out = tenferro::", stringify!($ctor), "(&a, &b).run()?;\n```")]
        pub fn $ctor<'a, T>(lhs: &'a AdTensor<T>, rhs: &'a AdTensor<T>) -> $builder<'a, T>
        where
            T: GenericAdRuntimeValue,
        {
            $builder { lhs, rhs }
        }
    };
    ($builder:ident, $ctor:ident, $doc_op:literal, real, |$self_ident:ident| $run:block) => {
        #[doc = concat!("Builder for AD `", $doc_op, "` on tensors.")]
        #[doc = ""]
        #[doc = "# Examples"]
        #[doc = ""]
        #[doc = concat!("```ignore\nlet out = tenferro::", stringify!($ctor), "(&a, &b).run()?;\n```")]
        pub struct $builder<'a, T: Scalar> {
            lhs: &'a AdTensor<T>,
            rhs: &'a AdTensor<T>,
        }

        impl<'a, T> $builder<'a, T>
        where
            T: RealAdRuntimeValue,
        {
            #[doc = concat!("Executes AD `", $doc_op, "`.")]
            #[doc = ""]
            #[doc = "# Examples"]
            #[doc = ""]
            #[doc = "```ignore\nlet out = builder.run()?;\n```"]
            pub fn run(self) -> Result<AdTensor<T>> {
                let $self_ident = self;
                $run
            }
        }

        #[doc = concat!("Creates a builder for AD `", $doc_op, "`.")]
        #[doc = ""]
        #[doc = "# Examples"]
        #[doc = ""]
        #[doc = concat!("```ignore\nlet out = tenferro::", stringify!($ctor), "(&a, &b).run()?;\n```")]
        pub fn $ctor<'a, T>(lhs: &'a AdTensor<T>, rhs: &'a AdTensor<T>) -> $builder<'a, T>
        where
            T: RealAdRuntimeValue,
        {
            $builder { lhs, rhs }
        }
    };
}

pub(super) use define_binary_ad_builder;
