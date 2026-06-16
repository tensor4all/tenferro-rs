//! Plotting helpers for the KdV PINN sample, using the `plotters` crate.

use plotters::prelude::*;
use std::error::Error;

/// Write an animated GIF comparing the analytic and predicted solution curves.
///
/// `xs` is the spatial grid. `frames` is a slice of `(t, analytic, predicted)`
/// values at each animation frame. The GIF uses 640×480 pixels with a delay of
/// 100 ms per frame.
pub(crate) fn write_comparison_gif(
    path: &str,
    xs: &[f64],
    frames: &[(f64, Vec<f64>, Vec<f64>)],
) -> Result<(), Box<dyn Error>> {
    const W: u32 = 640;
    const H: u32 = 480;

    let backend = BitMapBackend::gif(path, (W, H), 100)?;
    let root = backend.into_drawing_area();

    for (t, analytic, predicted) in frames {
        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .caption(format!("KdV soliton at t = {:.2}", t), ("sans-serif", 24))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(50)
            .build_cartesian_2d(-5.0..5.0, -0.5..2.5)?;

        chart
            .configure_mesh()
            .x_labels(11)
            .y_labels(6)
            .x_desc("x")
            .y_desc("u")
            .draw()?;

        chart
            .draw_series(LineSeries::new(
                xs.iter().zip(analytic.iter()).map(|(&x, &u)| (x, u)),
                BLUE.stroke_width(2),
            ))?
            .label("analytic")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE.stroke_width(2)));

        chart
            .draw_series(LineSeries::new(
                xs.iter().zip(predicted.iter()).map(|(&x, &u)| (x, u)),
                RED.stroke_width(2),
            ))?
            .label("predicted")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED.stroke_width(2)));

        chart
            .configure_series_labels()
            .background_style(WHITE.mix(0.8))
            .border_style(BLACK)
            .draw()?;

        root.present()?;
    }

    Ok(())
}
