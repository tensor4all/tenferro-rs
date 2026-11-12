tenferro_extension_macros::api_ops! {
    family: f;
    layer concrete {
        trait_doc: "d";
        trait_def: pub trait T;
        impl_def: impl T for u8;
        receiver: self;
        tensor_arg: &u8;
        result: R;
        forward: f;
    }
    layer eager { describe_only; }
    op o {
        summary: "s"; operands: (a); options: O; output: V; dtypes: "d";
        layers: [concrete]; into: []; layout: "l"; errors: []; omissions: [];
        example concrete: "x";
        signature eager: "pub fn f::S::o(&mut self)";
    }
}

fn main() {}
