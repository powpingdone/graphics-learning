#![recursion_limit = "128"]
use image::Rgb;
use obj::Obj;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelRefMutIterator;
use rayon::iter::ParallelIterator;

#[allow(dead_code)]
mod vec;
use crate::vec::*;

fn bay_tri_area(a: &Vec3, b: &Vec3, c: &Vec3) -> f32 {
    0.5 * ((b[Y] - a[Y]) * (b[X] + a[X])
        + (c[Y] - b[Y]) * (c[X] + b[X])
        + (a[Y] - c[Y]) * (a[X] + c[X]))
}

fn project_perspective(vec: Vec3) -> Vec3 {
    const CAMERA: f32 = 5.;
    vec / (1. - (vec[Z] / CAMERA))
}

fn rot3(vec: Vec3, rot: Vec3) -> Vec3 {
    let (alpha, beta, gamma) = (rot[Z], rot[Y], rot[X]);
    let (acos, bcos, gcos) = (alpha.cos(), beta.cos(), gamma.cos());
    let (asin, bsin, gsin) = (alpha.sin(), beta.sin(), gamma.sin());
    let mut rot = Mat3::default();
    // from https://en.wikipedia.org/wiki/Rotation_matrix#General_3D_rotations
    rot[0][0] = acos * bcos;
    rot[1][0] = acos * bsin * gsin - asin * gcos;
    rot[2][0] = acos * bsin * gcos + asin * gsin;
    rot[0][1] = asin * bcos;
    rot[1][1] = asin * bsin * gsin + acos * gcos;
    rot[2][1] = asin * bsin * gcos - acos * gsin;
    rot[0][2] = -bsin;
    rot[1][2] = bcos * gsin;
    rot[2][2] = bcos * gcos;
    rot * vec
}

fn filled_triangle(
    img: &mut image::RgbImage,
    zbuf: &mut Vec<f32>,
    pt0: &Vec3,
    pt1: &Vec3,
    pt2: &Vec3,
) {
    // if area is too big
    let max_area = bay_tri_area(pt0, pt1, pt2);
    if max_area < 1.0 {
        return;
    }

    // make max bounding box
    let min_x = (pt0[X].min(pt1[X].min(pt2[X])) as u32).max(0);
    let min_y = (pt0[Y].min(pt1[Y].min(pt2[Y])) as u32).max(0);
    let max_x = (pt0[X].max(pt1[X].max(pt2[X])) as u32 + 1).min(img.width());
    let max_y = (pt0[Y].max(pt1[Y].max(pt2[Y])) as u32 + 1).min(img.height());

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
            let px_gen = Vec3::from([x as f32, y as f32, 0.0]);
            let a = bay_tri_area(&px_gen, pt1, pt2) / max_area;
            let b = bay_tri_area(&px_gen, pt2, pt0) / max_area;
            let c = bay_tri_area(&px_gen, pt0, pt1) / max_area;
            if a < 0.0 || b < 0.0 || c < 0.0 {
                return;
            }

            // depth buffer check
            let depth = (a * pt0[Z]) + (b * pt1[Z]) + (c * pt2[Z]);
            if depth <= *depth_px {
                return;
            }

            // update depth and pixel
            *depth_px = depth;
            *px = Rgb(tri_color);
        });
}

// https://haqr.eu/tinyrenderer
fn main() {
    let mut img = image::RgbImage::new(1024, 1024);
    let (width, height) = img.dimensions();
    let mut zbuf = vec![0f32; height as usize * width as usize];
    let width = width - 1;
    let height = height - 1;

    // extract out required parts
    let obj = Obj::load("african_head.obj").unwrap().data;
    let faces = &obj.objects[0].groups[0].polys;
    let verts = &obj.position;
    let proj_x = |x: f32| (x + 1.0) * width as f32 / 2.;
    let proj_y = |y: f32| (y + 1.0) * height as f32 / 2.;
    let proj_z = |z: f32| (z + 1.0) * 255.0 / 2.;

    // per polygon
    for face in faces.iter() {
        // decode face positions
        let pts = [face.0[0].0, face.0[1].0, face.0[2].0];
        let [vert0, vert1, vert2] = [verts[pts[0]], verts[pts[1]], verts[pts[2]]];
        // pack into structs
        let pt0 = Vec3::from(vert0);
        let pt1 = Vec3::from(vert1);
        let pt2 = Vec3::from(vert2);
        // rot
        let roty = Vec3::from([0., 30f32.to_radians(), 0.]);
        let pt0 = rot3(pt0, roty);
        let pt1 = rot3(pt1, roty);
        let pt2 = rot3(pt2, roty);
        // perspective
        let pt0 = project_perspective(pt0);
        let pt1 = project_perspective(pt1);
        let pt2 = project_perspective(pt2);
        // adjust points to canvas
        let pt0 = Vec3::from([proj_x(pt0[X]), proj_y(pt0[Y]), proj_z(pt0[Z])]);
        let pt1 = Vec3::from([proj_x(pt1[X]), proj_y(pt1[Y]), proj_z(pt1[Z])]);
        let pt2 = Vec3::from([proj_x(pt2[X]), proj_y(pt2[Y]), proj_z(pt2[Z])]);
        // triangle
        filled_triangle(&mut img, &mut zbuf, &pt0, &pt1, &pt2);
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
