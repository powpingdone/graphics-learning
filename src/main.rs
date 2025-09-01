use image::Rgb;
use obj::Obj;
use rand::Rng;
use rayon::iter::ParallelIterator;

#[derive(Debug, Clone)]
struct ImgCoord2D {
    pub x: i32,
    pub y: i32,
}

impl ImgCoord2D {
    pub fn t(self) -> Self {
        Self {
            x: self.y,
            y: self.x,
        }
    }
}

fn bay_tri_area(a: &ImgCoord2D, b: &ImgCoord2D, c: &ImgCoord2D) -> f32 {
    let a = (a.x as f32, a.y as f32);
    let b = (b.x as f32, b.y as f32);
    let c = (c.x as f32, c.y as f32);
    0.5 * ((b.1 - a.1) * (b.0 + a.0) + (c.1 - b.1) * (c.0 + b.0) + (a.1 - c.1) * (a.0 + c.0))
}

fn filled_triangle(
    img: &mut image::RgbImage,
    pt0: &ImgCoord2D,
    pt1: &ImgCoord2D,
    pt2: &ImgCoord2D,
    color: &Rgb<u8>,
) {
    // make max bounding box
    let min_x = pt0.x.min(pt1.x.min(pt2.x)) as u32;
    let min_y = pt0.y.min(pt1.y.min(pt2.y)) as u32;
    let max_x = pt0.x.max(pt1.x.max(pt2.x)) as u32;
    let max_y = pt0.y.max(pt1.y.max(pt2.y)) as u32;
    let max_area = bay_tri_area(pt0, pt1, pt2);
    if max_area < 1.0 {
        return;
    }

    // run on bounding box
    img.par_enumerate_pixels_mut()
        .filter(|(x, y, _)| {
            // within bounding box
            *x >= min_x && *x < max_x && *y >= min_y && *y < max_y
        })
        .for_each(|(x, y, px)| {
            let a = bay_tri_area(
                &ImgCoord2D {
                    x: x as i32,
                    y: y as i32,
                },
                pt1,
                pt2,
            ) / max_area;
            let b = bay_tri_area(
                &ImgCoord2D {
                    x: x as i32,
                    y: y as i32,
                },
                pt2,
                pt0,
            ) / max_area;
            let c = bay_tri_area(
                &ImgCoord2D {
                    x: x as i32,
                    y: y as i32,
                },
                pt0,
                pt1,
            ) / max_area;

            // within triangle
            if a > 0.0 && b > 0.0 && c > 0.0 {
                // color
                *px = *color;
            }
        });
}

fn main() {
    let mut img = image::RgbImage::new(1024, 1024);
    let (width, height) = img.dimensions();
    let width = width - 1;
    let height = height - 1;

    // extract out required parts
    let obj = Obj::load("african_head.obj").unwrap().data;
    let faces = &obj.objects[0].groups[0].polys;
    let verts = &obj.position;
    let mut rando = rand::rng();

    // per polygon
    for face in faces.iter() {
        // decode face positions
        let pts = [face.0[0].0, face.0[1].0, face.0[2].0];
        let [vert0, vert1, vert2] = [verts[pts[0]], verts[pts[1]], verts[pts[2]]];
        // adjust points to canvas
        let pt0 = ImgCoord2D {
            x: ((vert0[0] + 1.0) * ((width as f32) / 2.0)).round() as i32,
            y: (height as i32 - ((vert0[1] + 1.0) * (height as f32) / 2.0).round() as i32),
        };
        let pt1 = ImgCoord2D {
            x: ((vert1[0] + 1.0) * ((width as f32) / 2.0)).round() as i32,
            y: (height as i32 - ((vert1[1] + 1.0) * ((height as f32) / 2.0)).round() as i32),
        };
        let pt3 = ImgCoord2D {
            x: ((vert2[0] + 1.0) * ((width as f32) / 2.0)).round() as i32,
            y: (height as i32 - ((vert2[1] + 1.0) * (height as f32) / 2.0).round() as i32),
        };
        // triangle
        filled_triangle(&mut img, &pt0, &pt1, &pt3, &Rgb(rando.random::<[u8; 3]>()));
    }

    img.save("arf.tga").unwrap();
}
