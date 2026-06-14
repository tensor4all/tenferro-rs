use tenferro_runtime::{GraphInstructionView, GraphProgramLoweringShapeError};

use crate::{Error, Result};

pub(crate) fn static_output_shape(
    inst: GraphInstructionView<'_>,
    output_index: usize,
    input_shapes: &[&[usize]],
) -> Result<Vec<usize>> {
    inst.static_output_shape(output_index, input_shapes)
        .map_err(|err| match err {
            GraphProgramLoweringShapeError::MissingOutput { op, output_index } => {
                Error::InvalidProgram {
                    message: format!(
                        "ExecOp::{op} missing output_extents for output {output_index}"
                    ),
                }
            }
            GraphProgramLoweringShapeError::NonStatic {
                op,
                output_index,
                axis,
                kind,
            } => Error::NonStaticShape {
                op,
                output_index,
                axis,
                kind,
            },
        })
}
