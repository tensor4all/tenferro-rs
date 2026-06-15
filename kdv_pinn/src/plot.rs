//! Minimal 2-D curve plotting helpers for generating comparison images.

use image::buffer::ConvertBuffer;
use image::{Rgb, RgbImage};

const WIDTH: u32 = 640;
const HEIGHT: u32 = 480;
const LEFT: u32 = 60;
const RIGHT: u32 = 20;
const TOP: u32 = 20;
const BOTTOM: u32 = 50;

const COLOR_TRUE: [u8; 3] = [0, 0, 255]; // blue
const COLOR_PRED: [u8; 3] = [255, 0, 0]; // red
const COLOR_AXIS: [u8; 3] = [0, 0, 0]; // black
const COLOR_BG: [u8; 3] = [255, 255, 255]; // white

/// Create a blank white image with the standard plot size.
pub(crate) fn new_image() -> RgbImage {
    RgbImage::from_pixel(WIDTH, HEIGHT, Rgb(COLOR_BG))
}

/// Draw a straight line between two pixel coordinates with the given color.
fn draw_line(img: &mut RgbImage, mut x0: i32, mut y0: i32, x1: i32, y1: i32, color: [u8; 3]) {
    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;

    loop {
        if x0 >= 0 && x0 < WIDTH as i32 && y0 >= 0 && y0 < HEIGHT as i32 {
            img.put_pixel(x0 as u32, y0 as u32, Rgb(color));
        }
        if x0 == x1 && y0 == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x0 += sx;
        }
        if e2 <= dx {
            err += dx;
            y0 += sy;
        }
    }
}

/// Map a data point `(x, u)` to pixel coordinates.
fn map(x: f64, u: f64, u_min: f64, u_max: f64) -> (i32, i32) {
    let plot_width = (WIDTH - LEFT - RIGHT) as f64;
    let plot_height = (HEIGHT - TOP - BOTTOM) as f64;
    let px = LEFT as f64 + (x + 5.0) / 10.0 * plot_width;
    let py = (HEIGHT - BOTTOM) as f64 - (u - u_min) / (u_max - u_min) * plot_height;
    (px as i32, py as i32)
}

/// Draw x- and u-axes into the image using the current vertical range.
pub(crate) fn draw_axes(img: &mut RgbImage, u_min: f64, u_max: f64) {
    let left = map(-5.0, 0.0, u_min, u_max);
    let right = map(5.0, 0.0, u_min, u_max);
    draw_line(img, left.0, left.1, right.0, right.1, COLOR_AXIS);

    let bottom = (HEIGHT - BOTTOM) as i32;
    let x_axis_x = map(0.0, u_min, u_min, u_max).0;
    draw_line(img, x_axis_x, TOP as i32, x_axis_x, bottom, COLOR_AXIS);
}

/// Draw a curve given by `(xs, us)` and an explicit vertical range.
pub(crate) fn draw_curve(
    img: &mut RgbImage,
    xs: &[f64],
    us: &[f64],
    u_min: f64,
    u_max: f64,
    color: [u8; 3],
) {
    assert_eq!(xs.len(), us.len());
    if xs.len() < 2 {
        return;
    }
    let p0 = map(xs[0], us[0], u_min, u_max);
    for i in 1..xs.len() {
        let p1 = map(xs[i], us[i], u_min, u_max);
        draw_line(img, p0.0, p0.1, p1.0, p1.1, color);
    }
}

/// Draw a comparison frame showing the true and predicted solution curves.
///
/// `xs` must be the spatial grid, `truth` and `pred` the corresponding values.
/// The vertical axis is fixed to `[-0.5, 2.5]` so that all frames share the same
/// scale.
pub(crate) fn draw_comparison_frame(xs: &[f64], truth: &[f64], pred: &[f64]) -> RgbImage {
    let u_min = -0.5;
    let u_max = 2.5;
    let mut img = new_image();
    draw_axes(&mut img, u_min, u_max);
    draw_curve(&mut img, xs, truth, u_min, u_max, COLOR_TRUE);
    draw_curve(&mut img, xs, pred, u_min, u_max, COLOR_PRED);
    img
}

/// Encode a sequence of RGB images as an animated GIF.
pub(crate) fn encode_gif(
    out_path: &str,
    frames: &[RgbImage],
    delay_hundredths: u16,
) -> Result<(), image::ImageError> {
    use image::codecs::gif::GifEncoder;
    use image::{Delay, Frame};
    use std::fs::File;

    let out = File::create(out_path)?;
    let mut encoder = GifEncoder::new(out);
    let delay = Delay::from_numer_denom_ms(delay_hundredths as u32 * 10, 1);
    for img in frames {
        let rgba: image::RgbaImage = img.convert();
        let frame = Frame::from_parts(rgba, 0, 0, delay);
        encoder.encode_frame(frame)?;
    }
    Ok(())
}
