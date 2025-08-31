use image::Rgb;
use obj::Obj;

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

fn line<Img, Pixel: Clone>(img: &mut Img, pt0: &ImgCoord2D, pt1: &ImgCoord2D, color: &Pixel)
where
    Img: image::GenericImage<Pixel = Pixel>,
{
    let dims = img.dimensions();

    // check if transpose due to steep lines
    let (pt0, pt1, transpose) = if (pt0.x - pt1.x).abs() < (pt0.y - pt1.y).abs() {
        (pt0.clone().t(), pt1.clone().t(), true)
    } else {
        (pt0.clone(), pt1.clone(), false)
    };

    // order lower x first
    let (pt0, pt1) = if pt0.x <= pt1.x {
        (pt0, pt1)
    } else {
        (pt1, pt0)
    };

    // setup delta
    let delta_x = pt1.x - pt0.x;
    let delta_err = (pt1.y - pt0.y).abs() * 2;
    let mut err = 0;
    let mut y_at = pt0.y;
    let mut place_px_fn: Box<dyn FnMut(u32, u32)> = if transpose {
        Box::new(|y, x| img.put_pixel(x, y, color.clone()))
    } else {
        Box::new(|x, y| img.put_pixel(x, y, color.clone()))
    };

    // put pixels
    for x_at in (pt0.x)..=(pt1.x) {
        // skip outside the image
        if y_at < 0 || x_at < 0 || x_at as u32 > dims.0 || y_at as u32 > dims.1 {
            continue;
        }

        // pixel place
        place_px_fn(x_at as u32, y_at as u32);

        // compute error
        err += delta_err;
        if err > delta_x {
            // on significant err, move pixel
            if pt0.y < pt1.y {
                y_at += 1
            } else {
                y_at -= 1
            };
            err -= delta_x * 2;
        }
    }
}

fn triangle<Img, Pixel: Clone>(
    img: &mut Img,
    pt0: &ImgCoord2D,
    pt1: &ImgCoord2D,
    pt2: &ImgCoord2D,
    color: &Pixel,
) where
    Img: image::GenericImage<Pixel = Pixel>,
{
    line(img, &pt0, &pt1, &color);
    line(img, &pt1, &pt2, &color);
    line(img, &pt2, &pt0, &color);
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
        triangle(&mut img, &pt0, &pt1, &pt3, &Rgb([255, 0, 0]));
    }

    img.save("arf.tga").unwrap();
}
