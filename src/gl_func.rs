use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelRefMutIterator;
use rayon::iter::ParallelIterator;

use crate::vec::*;

pub trait FragShader {
    fn fragment(&self, frag_coord: Vec3) -> Option<Vec3>;
}

// drawing context that is given to the user
pub struct OwnedDrawingContext {
    frame: image::RgbImage,
    zbuf: Vec<f32>,
    viewport: Mat4,
}

// drawing context that this lib draws to, borrowing from the prev struct
pub struct MutatingDrawingContext<'a> {
    frame: &'a mut image::RgbImage,
    zbuf: &'a mut Vec<f32>,
    viewport: &'a Mat4,
}

impl OwnedDrawingContext {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            viewport: viewport_mat(
                width as f32 * 7. / 8.,
                height as f32 * 7. / 8.,
                width as f32 / 16.,
                height as f32 / 16.,
            ),
            frame: image::RgbImage::new(width, height),
            zbuf: vec![-f32::INFINITY; height as usize * width as usize],
        }
    }

    pub fn mutate(&'_ mut self) -> MutatingDrawingContext<'_> {
        MutatingDrawingContext {
            frame: &mut self.frame,
            zbuf: &mut self.zbuf,
            viewport: &self.viewport,
        }
    }

    pub fn clear_bufs(&mut self) {
        self.frame.fill(0);
        self.zbuf.fill(0.);
    }

    pub fn copy_frame(&self) -> image::RgbImage {
        self.frame.clone()
    }

    pub fn render_zbuf(&self) -> image::GrayImage {
        let zmin = self.zbuf.iter().fold(f32::INFINITY, |acc, x| {
            if x.is_finite() && *x < acc { *x } else { acc }
        });
        let zmax = self
            .zbuf
            .iter()
            .fold(-f32::INFINITY, |acc, x| if *x > acc { *x } else { acc });
        image::GrayImage::from_vec(
            self.frame.width(),
            self.frame.height(),
            self.zbuf
                .iter()
                .map(|x| {
                    if x.is_infinite() {
                        0
                    } else {
                        255 - (((zmax - x) / (zmax - zmin)) * 255.0) as u8
                    }
                })
                .collect(),
        )
        .unwrap()
    }
}

// transform matrix for coordinates on the framebuffer
pub fn viewport_mat(width: f32, height: f32, x: f32, y: f32) -> Mat4 {
    let mut i = Mat4::identity();
    i[0][X] = width / 2.;
    i[0][W] = x + width / 2.;
    i[1][Y] = height / 2.;
    i[1][W] = y + height / 2.;
    i
}

// perspective distortion
pub fn perpsective_mat(focal_length: f32) -> Mat4 {
    let mut i = Mat4::identity();
    i[3][Z] = -1. / focal_length;
    i
}

// model transform coordinates, aka LookAt
pub fn modelview_mat(eye: Vec3, center: Vec3, up: Vec3) -> Mat4 {
    let n = (eye - center).normalize();
    let l = up.cross(n).normalize();
    let m = n.cross(l).normalize();
    Mat4::from([
        l.extend(0.),
        m.extend(0.),
        n.extend(0.),
        Vec4::from([0., 0., 0., 1.]),
    ]) * Mat4::from([
        [1., 0., 0., -center[X]],
        [0., 1., 0., -center[Y]],
        [0., 0., 1., -center[Z]],
        [0., 0., 0., 1.],
    ])
}

pub fn rasterize(
    ctx: MutatingDrawingContext<'_>,
    shader: &(dyn FragShader + Send + Sync),
    tri: Mat3x4,
) {
    // normalized device coords
    let ndc: Mat3x4 = tri.apply(|i| i / i[W]).into();
    // screeen coords
    let screen: Mat3x2 = ndc.apply(|i| (*ctx.viewport * i).swiz(X | Y)).into();

    // if tri is too small or a backface bail
    let abc = Mat3::from(screen.apply(|i| i.extend(1.)));
    if abc.det() < 1.0 {
        return;
    }

    // make bounding box
    let min_x = screen
        .iter()
        .fold(f32::INFINITY, |acc, i| acc.min(i[X]))
        .max(0.)
        .floor() as u32;
    let min_y = screen
        .iter()
        .fold(f32::INFINITY, |acc, i| acc.min(i[Y]))
        .max(0.)
        .floor() as u32;
    let max_x = screen
        .iter()
        .fold(-f32::INFINITY, |acc, i| acc.max(i[X]))
        .min(ctx.frame.width() as f32 - 1.)
        .ceil() as u32;
    let max_y = screen
        .iter()
        .fold(-f32::INFINITY, |acc, i| acc.max(i[Y]))
        .min(ctx.frame.height() as f32 - 1.)
        .ceil() as u32;

    // run on frame
    ctx.frame
        .par_enumerate_pixels_mut()
        // with zbuffer
        .zip(ctx.zbuf.par_iter_mut())
        // within bounding box
        .filter(|((x, y, _), _)| *x >= min_x && *x < max_x && *y >= min_y && *y < max_y)
        .for_each(|((x, y, px), depth_px)| {
            // within triangle
            let bc = abc.invert_t() * Vec3::from([x as f32, y as f32, 1.]);
            if bc.iter().any(|x| *x < 0.) {
                return;
            }

            // highest pixel, according to the depth buffer
            let depth = bc.dot(ndc.apply(|i| i[Z]).into());
            if depth <= *depth_px {
                return;
            }

            // if shader returns a color
            if let Some(rgb) = shader.fragment(bc) {
                // convert to u8 rgb
                let rgb = image::Rgb([
                    rgb[R].clamp(0., 255.) as u8,
                    rgb[G].clamp(0., 255.) as u8,
                    rgb[B].clamp(0., 255.) as u8,
                ]);
                // then update depth and pixel
                *depth_px = depth;
                *px = rgb;
            }
        });
}
