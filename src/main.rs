#![recursion_limit = "128"]
use image::Rgb;
use obj::Obj;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelRefMutIterator;
use rayon::iter::ParallelIterator;

#[allow(dead_code)]
mod vec;
use crate::vec::*;

fn filled_triangle(
    img: &mut image::RgbImage,
    zbuf: &mut Vec<f32>,
    clip0: Vec4,
    clip1: Vec4,
    clip2: Vec4,
    viewport: Mat4,
) {
    let [ndc0, ndc1, ndc2] = [clip0 / clip0[W], clip1 / clip1[W], clip2 / clip2[W]];
    let [idp0, idp1, idp2] = [viewport * ndc0, viewport * ndc1, viewport * ndc2];
    let [screen0, screen1, screen2] = [
        Vec2::from([idp0[X], idp0[Y]]),
        Vec2::from([idp1[X], idp1[Y]]),
        Vec2::from([idp2[X], idp2[Y]]),
    ];

    // if area is too small
    let abc = Mat3::from([
        [screen0[X], screen0[Y], 1.],
        [screen1[X], screen1[Y], 1.],
        [screen2[X], screen2[Y], 1.],
    ]);
    if abc.det() < 1.0 {
        return;
    }

    // make max bounding box
    let min_x = screen0[X].min(screen1[X]).min(screen2[X]).max(0.).round() as u32;
    let min_y = screen0[Y].min(screen1[Y]).min(screen2[Y]).max(0.).round() as u32;
    let max_x = screen0[X]
        .max(screen1[X])
        .max(screen2[X])
        .min(img.width() as f32 - 1.)
        .round() as u32;
    let max_y = screen0[Y]
        .max(screen1[Y])
        .max(screen2[Y])
        .min(img.height() as f32 - 1.)
        .round() as u32;

    let tri_color = rand::random();

    // run on bounding box
    img.par_enumerate_pixels_mut()
        // with zbuffer
        .zip(zbuf.par_iter_mut())
        .filter(|((x, y, _), _)| {
            // within bounding box
            *x >= min_x && *x < max_x && *y >= min_y && *y < max_y
        })
        .for_each(|((x, y, px), depth_px)| {
            // within triangle
            let bc = abc.invert_t() * Vec3::from([x as f32, y as f32, 1.]);
            if bc.iter().any(|x| *x < 0.) {
                return;
            }

            // depth buffer check
            let depth = bc.dot(Vec3::from([ndc0[Z], ndc1[Z], ndc2[Z]]));
            if depth <= *depth_px {
                return;
            }

            // update depth and pixel
            *depth_px = depth;
            *px = Rgb(tri_color);
        });
}

fn viewport_mat(width: f32, height: f32, x: f32, y: f32) -> Mat4 {
    let mut i = Mat4::identity();
    i[0][X] = width / 2.;
    i[0][W] = x + width / 2.;
    i[1][Y] = height / 2.;
    i[1][W] = y + height / 2.;
    i
}

fn perpsective_mat(focal_length: f32) -> Mat4 {
    let mut i = Mat4::identity();
    i[3][Z] = -1. / focal_length;
    i
}

fn modelview_mat(eye: Vec3, center: Vec3, up: Vec3) -> Mat4 {
    let n = (eye - center).normalize();
    let l = up.cross(n).normalize();
    let m = n.cross(l).normalize();
    Mat4::from([
        [l[X], l[Y], l[Z], 0.],
        [m[X], m[Y], m[Z], 0.],
        [n[X], n[Y], n[Z], 0.],
        [0., 0., 0., 1.],
    ]) * Mat4::from([
        [1., 0., 0., -center[X]],
        [0., 1., 0., -center[Y]],
        [0., 0., 1., -center[Z]],
        [0., 0., 0., 1.],
    ])
}

// https://haqr.eu/tinyrenderer
fn main() {
    let mut img = image::RgbImage::new(1024, 1024);
    let (width, height) = img.dimensions();
    let mut zbuf = vec![0f32; height as usize * width as usize];
    let width = width - 1;
    let height = height - 1;
    let eye = Vec3::from([-1., 0., 2.]);
    let center = Vec3::from([0., 0., 0.]);
    let up = Vec3::from([0., 1., 0.]);
    let perspective = perpsective_mat((eye - center).norm());
    let modelview = modelview_mat(eye, center, up);
    let viewport = viewport_mat(
        width as f32 / 16.,
        height as f32 / 16.,
        width as f32 * 7. / 8.,
        height as f32 * 7. / 8.,
    );

    // extract out required parts
    let obj = Obj::load("african_head.obj").unwrap().data;
    let faces = &obj.objects[0].groups[0].polys;
    let verts = &obj.position;

    // per polygon
    for face in faces.iter() {
        // decode face positions
        let pts = [face.0[0].0, face.0[1].0, face.0[2].0];
        let [vert0, vert1, vert2] = [verts[pts[0]], verts[pts[1]], verts[pts[2]]];
        // pack into structs
        let pt0 = Vec3::from(vert0);
        let pt1 = Vec3::from(vert1);
        let pt2 = Vec3::from(vert2);
        // make clip coords
        let [clip0, clip1, clip2] = [
            perspective * modelview * Vec4::from([pt0[X], pt0[Y], pt0[Z], 1.]),
            perspective * modelview * Vec4::from([pt1[X], pt1[Y], pt1[Z], 1.]),
            perspective * modelview * Vec4::from([pt2[X], pt2[Y], pt2[Z], 1.]),
        ]; // triangle
        filled_triangle(&mut img, &mut zbuf, clip0, clip1, clip2, viewport);
    }

    // show output
    img.save("arf.tga").unwrap();

    // show zbuf
    let zmin = zbuf
        .iter()
        .fold(f32::INFINITY, |x, acc| if x < *acc { x } else { *acc });
    let zmax = zbuf
        .iter()
        .fold(-f32::INFINITY, |x, acc| if x > *acc { x } else { *acc });
    image::GrayImage::from_vec(
        width + 1,
        height + 1,
        zbuf.into_iter()
            .map(|x| 255 - (((zmax - x) / (zmax - zmin)) * 255.0) as u8)
            .collect(),
    )
    .unwrap()
    .save("zbuf.tga")
    .unwrap();
}
