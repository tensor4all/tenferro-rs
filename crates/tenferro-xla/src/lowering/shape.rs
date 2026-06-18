use tenferro_runtime::{GraphInstructionView, GraphProgramLoweringShapeError};

use crate::{Error, Result};

pub(crate) fn static_output_shape(
    inst: GraphInstructionView<'_>,
    output_index: usize,
    input_shapes: &[&[usize]],
) -> Result<Vec<usize>> {
    match inst.static_output_shape(output_index, input_shapes) {
        Ok(shape) => Ok(shape),
        Err(GraphProgramLoweringShapeError::MissingOutput { op, output_index }) => {
            Err(Error::InvalidProgram {
                message: format!("ExecOp::{op} missing output_extents for output {output_index}"),
            })
        }
        Err(GraphProgramLoweringShapeError::NonStatic {
            op,
            output_index,
            axis,
            kind,
        }) => Err(Error::NonStaticShape {
            op,
            output_index,
            axis,
            kind,
        }),
        Err(GraphProgramLoweringShapeError::InvalidDimExpr {
            op,
            output_index,
            axis,
            source,
        }) => Err(Error::InvalidProgram {
            message: format!(
                "ExecOp::{op} output {output_index} axis {axis} has invalid dimension expression: {source}"
            ),
        }),
    }
}

#[cfg(test)]
mod tests;
