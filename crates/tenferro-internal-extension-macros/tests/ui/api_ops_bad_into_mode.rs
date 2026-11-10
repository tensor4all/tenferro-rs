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
    op o {
        summary: "s"; operands: (a); options: O; output: V; dtypes: "d";
        layers: [concrete]; into: [sideways]; layout: "l"; errors: []; omissions: [];
        example concrete: "x";
    }
}

fn main() {}
