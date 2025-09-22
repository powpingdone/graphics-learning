use image::Rgb;
use obj::Obj;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelRefMutIterator;
use rayon::iter::ParallelIterator;

mod vec;
use crate::vec::*;

struct V3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

fn bay_tri_area(a: &V3, b: &V3, c: &V3) -> f32 {
    0.5 * ((b.y - a.y) * (b.x + a.x) + (c.y - b.y) * (c.x + b.x) + (a.y - c.y) * (a.x + c.x))
}

fn filled_triangle(img: &mut image::RgbImage, zbuf: &mut Vec<f32>, pt0: &V3, pt1: &V3, pt2: &V3) {
    // make max bounding box
    let min_x = pt0.x.min(pt1.x.min(pt2.x)) as u32;
    let min_y = pt0.y.min(pt1.y.min(pt2.y)) as u32;
    let max_x = pt0.x.max(pt1.x.max(pt2.x)) as u32 + 1;
    let max_y = pt0.y.max(pt1.y.max(pt2.y)) as u32 + 1;
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
            let px_gen = V3 {
                x: x as f32,
                y: y as f32,
                z: 0.0,
            };
            let a = bay_tri_area(&px_gen, pt1, pt2) / max_area;
            let b = bay_tri_area(&px_gen, pt2, pt0) / max_area;
            let c = bay_tri_area(&px_gen, pt0, pt1) / max_area;

            // depth buffer check
            let depth = (a * pt0.z) + (b * pt1.z) + (c * pt2.z);

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
        let pt0 = V3 {
            x: proj_x(vert0[0]),
            y: proj_y(vert0[1]),
            z: proj_z(vert0[2]),
        };
        let pt1 = V3 {
            x: proj_x(vert1[0]),
            y: proj_y(vert1[1]),
            z: proj_z(vert1[2]),
        };
        let pt2 = V3 {
            x: proj_x(vert2[0]),
            y: proj_y(vert2[1]),
            z: proj_z(vert2[2]),
        };
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
