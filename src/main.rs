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

fn filled_triangle(
    img: &mut image::RgbImage,
    zbuf: &mut Vec<f32>,
    pt0: &Vec3,
    pt1: &Vec3,
    pt2: &Vec3,
) {
    // make max bounding box
    let min_x = pt0[X].min(pt1[X].min(pt2[X])) as u32;
    let min_y = pt0[Y].min(pt1[Y].min(pt2[Y])) as u32;
    let max_x = pt0[X].max(pt1[X].max(pt2[X])) as u32 + 1;
    let max_y = pt0[Y].max(pt1[Y].max(pt2[Y])) as u32 + 1;
    let max_area = bay_tri_area(pt0, pt1, pt2);
    if max_area < 1.0 {
        return;
    }

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
            let px_gen = Vec3::from([x as f32, y as f32, 0.0]);
            let a = bay_tri_area(&px_gen, pt1, pt2) / max_area;
            let b = bay_tri_area(&px_gen, pt2, pt0) / max_area;
            let c = bay_tri_area(&px_gen, pt0, pt1) / max_area;

            // depth buffer check
            let depth = (a * pt0[Z]) + (b * pt1[Z]) + (c * pt2[Z]);

            // depth check and within triangle
            if depth > *depth_px && a > 0.0 && b > 0.0 && c > 0.0 {
                // update depth and pixel
                *depth_px = depth;
                *px = Rgb(tri_color);
            }
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
    let proj_x = |x: f32| (-x + 1.0) * width as f32 / 2.;
    let proj_y = |y: f32| (-y + 1.0) * height as f32 / 2.;
    let proj_z = |z: f32| (z + 1.0) * 255.0 / 2.;

    // per polygon
    for face in faces.iter() {
        // decode face positions
        let pts = [face.0[0].0, face.0[1].0, face.0[2].0];
        let [vert0, vert1, vert2] = [verts[pts[0]], verts[pts[1]], verts[pts[2]]];
        // adjust points to canvas, pack into coords struct
        let pt0 = Vec3::from([proj_x(vert0[0]), proj_y(vert0[1]), proj_z(vert0[2])]);
        let pt1 = Vec3::from([proj_x(vert1[0]), proj_y(vert1[1]), proj_z(vert1[2])]);
        let pt2 = Vec3::from([proj_x(vert2[0]), proj_y(vert2[1]), proj_z(vert2[2])]);
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
