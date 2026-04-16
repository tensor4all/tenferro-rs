use cubecl::prelude::*;

use super::{axis_in_sequence, flat_to_multi_index, multi_to_flat_index, shape_product};

macro_rules! reduction_kernel {
    ($float_name:ident, $complex_name:ident, $op:tt) => {
        #[cube(launch_unchecked)]
        pub(crate) fn $float_name<F: Float>(
            out: &mut Array<F>,
            input: &Array<F>,
            #[comptime] input_shape: Sequence<usize>,
            #[comptime] output_shape: Sequence<usize>,
            #[comptime] axes: Sequence<usize>,
            #[comptime] reduction_shape: Sequence<usize>,
        ) {
            if ABSOLUTE_POS < out.len() {
                let out_idx = flat_to_multi_index(ABSOLUTE_POS, output_shape.clone());
                let reduce_elems = shape_product(reduction_shape.clone());
                let rank = input_shape.len();
                let reduce_rank = reduction_shape.len();
                let mut input_idx = Array::<usize>::new(rank);
                let mut red_idx = Array::<usize>::new(reduce_rank);
                let mut out_axis = 0usize;
                let mut red_axis = 0usize;
                #[unroll]
                for axis in 0..rank {
                    if axis_in_sequence(axes.clone(), axis) {
                        input_idx[axis] = 0;
                        red_axis += 1;
                    } else {
                        input_idx[axis] = out_idx[out_axis];
                        out_axis += 1;
                    }
                }
                let mut acc = input[multi_to_flat_index(&input_idx, input_shape.clone())];
                for red_flat in 1..reduce_elems {
                    let next_red_idx = flat_to_multi_index(red_flat, reduction_shape.clone());
                    #[unroll]
                    for axis in 0..reduce_rank {
                        red_idx[axis] = next_red_idx[axis];
                    }
                    out_axis = 0;
                    red_axis = 0;
                    #[unroll]
                    for axis in 0..rank {
                        if axis_in_sequence(axes.clone(), axis) {
                            input_idx[axis] = red_idx[red_axis];
                            red_axis += 1;
                        } else {
                            input_idx[axis] = out_idx[out_axis];
                            out_axis += 1;
                        }
                    }
                    acc = acc $op input[multi_to_flat_index(&input_idx, input_shape.clone())];
                }
                out[ABSOLUTE_POS] = acc;
            }
        }

        #[cube(launch_unchecked)]
        pub(crate) fn $complex_name<C: Complex>(
            out: &mut Array<C>,
            input: &Array<C>,
            #[comptime] input_shape: Sequence<usize>,
            #[comptime] output_shape: Sequence<usize>,
            #[comptime] axes: Sequence<usize>,
            #[comptime] reduction_shape: Sequence<usize>,
        ) {
            if ABSOLUTE_POS < out.len() {
                let out_idx = flat_to_multi_index(ABSOLUTE_POS, output_shape.clone());
                let reduce_elems = shape_product(reduction_shape.clone());
                let rank = input_shape.len();
                let reduce_rank = reduction_shape.len();
                let mut input_idx = Array::<usize>::new(rank);
                let mut red_idx = Array::<usize>::new(reduce_rank);
                let mut out_axis = 0usize;
                let mut red_axis = 0usize;
                #[unroll]
                for axis in 0..rank {
                    if axis_in_sequence(axes.clone(), axis) {
                        input_idx[axis] = 0;
                        red_axis += 1;
                    } else {
                        input_idx[axis] = out_idx[out_axis];
                        out_axis += 1;
                    }
                }
                let mut acc = input[multi_to_flat_index(&input_idx, input_shape.clone())];
                for red_flat in 1..reduce_elems {
                    let next_red_idx = flat_to_multi_index(red_flat, reduction_shape.clone());
                    #[unroll]
                    for axis in 0..reduce_rank {
                        red_idx[axis] = next_red_idx[axis];
                    }
                    out_axis = 0;
                    red_axis = 0;
                    #[unroll]
                    for axis in 0..rank {
                        if axis_in_sequence(axes.clone(), axis) {
                            input_idx[axis] = red_idx[red_axis];
                            red_axis += 1;
                        } else {
                            input_idx[axis] = out_idx[out_axis];
                            out_axis += 1;
                        }
                    }
                    acc = acc $op input[multi_to_flat_index(&input_idx, input_shape.clone())];
                }
                out[ABSOLUTE_POS] = acc;
            }
        }
    };
}

reduction_kernel!(reduce_sum_float, reduce_sum_complex, +);
reduction_kernel!(reduce_prod_float, reduce_prod_complex, *);

#[cube(launch_unchecked)]
pub(crate) fn reduce_max_float<F: Float>(
    out: &mut Array<F>,
    input: &Array<F>,
    #[comptime] input_shape: Sequence<usize>,
    #[comptime] output_shape: Sequence<usize>,
    #[comptime] axes: Sequence<usize>,
    #[comptime] reduction_shape: Sequence<usize>,
) {
    if ABSOLUTE_POS < out.len() {
        let out_idx = flat_to_multi_index(ABSOLUTE_POS, output_shape.clone());
        let reduce_elems = shape_product(reduction_shape.clone());
        let rank = input_shape.len();
        let reduce_rank = reduction_shape.len();
        let mut input_idx = Array::<usize>::new(rank);
        let mut red_idx = Array::<usize>::new(reduce_rank);
        let mut out_axis = 0usize;
        let mut red_axis = 0usize;
        #[unroll]
        for axis in 0..rank {
            if axis_in_sequence(axes.clone(), axis) {
                input_idx[axis] = 0;
                red_axis += 1;
            } else {
                input_idx[axis] = out_idx[out_axis];
                out_axis += 1;
            }
        }
        let mut acc = input[multi_to_flat_index(&input_idx, input_shape.clone())];
        for red_flat in 1..reduce_elems {
            let next_red_idx = flat_to_multi_index(red_flat, reduction_shape.clone());
            #[unroll]
            for axis in 0..reduce_rank {
                red_idx[axis] = next_red_idx[axis];
            }
            out_axis = 0;
            red_axis = 0;
            #[unroll]
            for axis in 0..rank {
                if axis_in_sequence(axes.clone(), axis) {
                    input_idx[axis] = red_idx[red_axis];
                    red_axis += 1;
                } else {
                    input_idx[axis] = out_idx[out_axis];
                    out_axis += 1;
                }
            }
            acc = acc.max(input[multi_to_flat_index(&input_idx, input_shape.clone())]);
        }
        out[ABSOLUTE_POS] = acc;
    }
}

#[cube(launch_unchecked)]
pub(crate) fn reduce_min_float<F: Float>(
    out: &mut Array<F>,
    input: &Array<F>,
    #[comptime] input_shape: Sequence<usize>,
    #[comptime] output_shape: Sequence<usize>,
    #[comptime] axes: Sequence<usize>,
    #[comptime] reduction_shape: Sequence<usize>,
) {
    if ABSOLUTE_POS < out.len() {
        let out_idx = flat_to_multi_index(ABSOLUTE_POS, output_shape.clone());
        let reduce_elems = shape_product(reduction_shape.clone());
        let rank = input_shape.len();
        let reduce_rank = reduction_shape.len();
        let mut input_idx = Array::<usize>::new(rank);
        let mut red_idx = Array::<usize>::new(reduce_rank);
        let mut out_axis = 0usize;
        let mut red_axis = 0usize;
        #[unroll]
        for axis in 0..rank {
            if axis_in_sequence(axes.clone(), axis) {
                input_idx[axis] = 0;
                red_axis += 1;
            } else {
                input_idx[axis] = out_idx[out_axis];
                out_axis += 1;
            }
        }
        let mut acc = input[multi_to_flat_index(&input_idx, input_shape.clone())];
        for red_flat in 1..reduce_elems {
            let next_red_idx = flat_to_multi_index(red_flat, reduction_shape.clone());
            #[unroll]
            for axis in 0..reduce_rank {
                red_idx[axis] = next_red_idx[axis];
            }
            out_axis = 0;
            red_axis = 0;
            #[unroll]
            for axis in 0..rank {
                if axis_in_sequence(axes.clone(), axis) {
                    input_idx[axis] = red_idx[red_axis];
                    red_axis += 1;
                } else {
                    input_idx[axis] = out_idx[out_axis];
                    out_axis += 1;
                }
            }
            acc = acc.min(input[multi_to_flat_index(&input_idx, input_shape.clone())]);
        }
        out[ABSOLUTE_POS] = acc;
    }
}
