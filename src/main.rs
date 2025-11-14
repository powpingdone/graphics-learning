#![recursion_limit = "128"]
use std::ffi::OsStr;
use std::fs::File;
use std::io::BufReader;

use image::ImageReader;
use obj::Obj;

#[allow(dead_code)]
mod vec;
use crate::vec::*;
mod gl_func;
use crate::gl_func::*;

struct SmoothFragShaderWithModel {
    perspective: Mat4,
    modelview: Mat4,
    tri: Mat3,
    sun: Vec3,
    tri_normal: Option<Mat3>,
    nm_image: Option<image::RgbImage>,
    diffuse_image: Option<image::RgbImage>,
    specular_image: Option<image::GrayImage>,
    tri_texture: Option<Mat3x2>,
}

impl SmoothFragShaderWithModel {
    fn new(eye: Vec3, center: Vec3, up: Vec3, sun: Vec3) -> Self {
        Self {
            perspective: perpsective_mat((eye - center).norm()),
            modelview: modelview_mat(eye, center, up),
            tri: Mat3::default(),
            tri_normal: None,
            nm_image: None,
            tri_texture: None,
            diffuse_image: None,
            specular_image: None,
            sun,
        }
    }

    fn copy_tri_arr(&self) -> Mat3x4 {
        // extract out the homogenous triangle struct for the draw call
        self.tri.apply(|row| row.extend(1.)).into()
    }

    fn set_tri_from_coords(&mut self, pt0: Vec3, pt1: Vec3, pt2: Vec3) {
        self.tri =
            // warp each vertex through the model viewpoint and perspective
            Mat3x4::from([
                self.perspective * self.modelview * pt0.extend(1.),
                self.perspective * self.modelview * pt1.extend(1.),
                self.perspective * self.modelview * pt2.extend(1.),
            ])
            // remove normal coords, this is due to how self.tri is Mat3 for perf
            // reasons, as converting from mat3x4 to mat3 requires an extra div to
            // remove homogenous coords
            .apply(|row| (row / row[W]).shrink())
            .into();
    }

    fn set_tri_normal(&mut self, inp: Option<Mat3>) {
        self.tri_normal = inp;
    }

    fn set_tri_uv(&mut self, inp: Option<Mat3x2>) {
        self.tri_texture = inp;
    }

    fn set_nm_image(&mut self, inp: Option<image::RgbImage>) {
        self.nm_image = inp;
    }

    fn set_diffuse_image(&mut self, inp: Option<image::RgbImage>) {
        self.diffuse_image = inp;
    }

    fn set_specular_image(&mut self, inp: Option<image::GrayImage>) {
        self.specular_image = inp;
    }
}

// helper function to do scalar broadcast per point in each vector
#[inline]
fn map_frag_coord_and_matrix<T, R>(frag_coord: Vec3, matrix: T) -> R
where
    T: Iterator,
    <T as Iterator>::Item: std::ops::Mul<f32>,
    R: std::iter::Sum<<<T as std::iter::Iterator>::Item as std::ops::Mul<f32>>::Output>,
{
    // for each vec
    matrix
        // for each pt
        .zip(frag_coord.iter().copied())
        // mul vec * pt
        .map(|(e, i)| e * i)
        // sum to make vec
        .sum::<R>()
}

// helper function to map dims to uv image coords
fn get_uv_coord(frag_coord: Vec3, inp_coords: Mat3x2, dims: (u32, u32)) -> (u32, u32) {
    // find coords in the triangle
    let coord: Vec2 = map_frag_coord_and_matrix(frag_coord, inp_coords.into_iter());

    // convert coords to px indexes
    let x = (dims.0 as f32 * coord[X]).round().max(1.) as u32;
    let y = (dims.1 as f32 * coord[Y]).round().max(1.) as u32;
    (x - 1, y - 1)
}

impl FragShader for SmoothFragShaderWithModel {
    fn fragment(&self, frag_coord: Vec3) -> Option<Vec3> {
        // base color to be manipulated
        let model_color = if let Some(mat) = self.tri_texture
            && let Some(ref img) = self.diffuse_image
        {
            // pull from the diffuse image
            let (x, y) = get_uv_coord(frag_coord, mat, img.dimensions());
            let px = img
                .get_pixel(x, y)
                .0
                .into_iter()
                .map(|x| x as f32)
                .collect::<Box<[_]>>();
            Vec3::from([px[0], px[1], px[2]])
        } else {
            // set as a "notexture", which is alternating purple and black
            if frag_coord.into_iter().sum::<f32>() % 1. > 0.5 {
                Vec3::from([255., 0., 255.])
            } else {
                Vec3::from([0., 0., 0.])
            }
        };

        // the percentage of ambient lighting, the base of the model color
        let ambient = 0.5;

        // the texturing of the surface in relation to lighting
        let surface_norm = {
            if let Some(mat) = self.tri_texture
                && let Some(ref img) = self.nm_image
            {
                // use precomputed normal map
                let (x, y) = get_uv_coord(frag_coord, mat, img.dimensions());
                // fetch pixel and convert to [-1, 1]
                let px = img
                    .get_pixel(x, y)
                    .0
                    .to_owned()
                    .into_iter()
                    .map(|x| 2. * (x as f32 / 255.) - 1.)
                    .collect::<Box<[_]>>();
                // put into vec3
                Vec3::from([px[0], px[1], px[2]])
            } else if let Some(mat) = self.tri_normal {
                // compute surface normal if we have it and the normal map doesn't exist
                // vary the normals based on how far away the normal
                // is from the bary coord
                map_frag_coord_and_matrix::<_, Vec3>(frag_coord, mat.into_iter()).normalize()
            } else {
                // if we dont have any precomputed normals, compute the flat phong
                // get AB and AC vectors (or P0P1 and P0P2)
                let a = self.tri[1] - self.tri[0];
                let b = self.tri[2] - self.tri[0];
                // cross and make the normal vector for phong
                a.cross(b).normalize()
            }
        };

        // find how far away the angle is from the sun
        let diffuse = surface_norm.dot(self.sun).max(0.);
        // then, make opposite of the sun vector for specular
        let reflect = (surface_norm * surface_norm.dot(self.sun) * 2. - self.sun).normalize();

        // then see, in the z direction, how reflective the surface should be
        let specular = reflect[Z].max(0.).powf({
            if let Some(mat) = self.tri_texture
                && let Some(ref img) = self.specular_image
            {
                // use the specular map
                let (x, y) = get_uv_coord(frag_coord, mat, img.dimensions());
                let px = img.get_pixel(x, y).0;
                px[0] as f32
            } else {
                // default to "basically" no reflection
                0.
            }
        });

        // finally, the actual combining part to realize the pixel
        let final_model_color =
            model_color.apply(|i| i * (ambient + 0.5 * diffuse + 0.8 * specular).min(1.));
        Some(final_model_color)
    }
}

fn load_img(path: &OsStr, postfix: &str, text_ext: &OsStr) -> Option<image::RgbImage> {
    // generate path
    let mut path = std::path::PathBuf::from(path);
    let fname = std::path::PathBuf::from(path.file_name().unwrap());
    let mut prefix = fname.file_prefix().unwrap().to_os_string();
    prefix.push(format!("_{postfix}."));
    prefix.push(text_ext);
    path.set_file_name(prefix);
    println!("trying to load file: {}", path.display());
    // open file
    let reader = || -> Option<_> {
        // slight workaround to show file loading while still
        // allowing usage of `?`, just wrap in a closure and call it
        let file = File::open(path).ok()?;
        let buf_file = BufReader::new(file);
        ImageReader::with_format(
            buf_file,
            image::ImageFormat::from_extension(text_ext).unwrap(),
        )
        .decode()
        .ok()?
        .flipv()
        .fliph()
        .into_rgb8()
        .into()
    }();

    if reader.is_some() {
        println!("file loaded");
    } else {
        println!("could not load file");
    }
    reader
}

// https://haqr.eu/tinyrenderer
fn main() -> Result<(), usize> {
    let mut ctx = OwnedDrawingContext::new(1024, 1024);
    let eye = Vec3::from([1., 0., 2.]);
    let center = Vec3::from([0., 0., 0.]);
    // this is inverted due to how coords are upside down in image
    let up = Vec3::from([0., -1., 0.]);
    let sun = Vec3::from([1., 0., 1.]).normalize();
    let mut fragshader = SmoothFragShaderWithModel::new(eye, center, up, sun);

    // extract out arglist
    let mut args = std::env::args_os();
    drop(args.next()); // skip executable name
    let args = args.collect::<Vec<_>>();
    if args.len() == 0 {
        eprintln!("no args were provided, please provide paths to each .obj file");
        return Err(1);
    }
    if args.len() % 2 != 0 {
        eprintln!(
            "an uneven amount of arguments were provided, add a object file and the file extension that the images use"
        );
        return Err(1);
    }

    // per file path
    for combo in args.chunks_exact(2) {
        let path = &combo[0];
        let text_ext = &combo[1];

        // load object
        println!("loading object {}", path.display());
        let obj = Obj::load(path).unwrap().data;
        let verts = &obj.position;
        let norms = &obj.normal;
        let uv = &obj.texture;

        // load textures
        fragshader.set_nm_image(load_img(path, "nm", text_ext));
        fragshader.set_diffuse_image(load_img(path, "diffuse", text_ext));
        fragshader.set_specular_image(
            load_img(path, "spec", text_ext).map(|x| image::DynamicImage::from(x).into_luma8()),
        );

        // per object
        for object in &obj.objects {
            println!("rendering object {}", object.name);
            // per group of faces
            for group in &object.groups {
                let faces = &group.polys;
                println!(
                    "rendering object group \"{}\" with face count {}",
                    group.name,
                    faces.len()
                );

                // per polygon
                for face in faces.iter() {
                    // decode face positions
                    let pts = [face.0[0].0, face.0[1].0, face.0[2].0];
                    let [vert0, vert1, vert2] = [verts[pts[0]], verts[pts[1]], verts[pts[2]]];
                    // pack into structs
                    let pt0 = Vec3::from(vert0);
                    let pt1 = Vec3::from(vert1);
                    let pt2 = Vec3::from(vert2);
                    // make clip(triangle) coords
                    fragshader.set_tri_from_coords(pt0, pt1, pt2);
                    let tri = fragshader.copy_tri_arr();

                    // decode texture coords
                    let uv_pts = [face.0[0].1, face.0[1].1, face.0[2].1];
                    // if all verts have associated texture coords
                    if uv_pts.iter().all(|x| x.is_some()) {
                        // get associated pts
                        let uv_pts = uv_pts
                            .into_iter()
                            .map(|x| uv[x.unwrap()])
                            .collect::<Box<[_]>>();
                        // collect and set
                        let uv_verts = Mat3x2::from([uv_pts[0], uv_pts[1], uv_pts[2]]);
                        fragshader.set_tri_uv(Some(uv_verts));
                    } else {
                        // unset otherwise
                        fragshader.set_tri_uv(None);
                    }

                    // decode normals
                    let npts = [face.0[0].2, face.0[1].2, face.0[2].2];
                    // if all verts have associated normals
                    if npts.iter().all(|x| x.is_some()) {
                        // get associated pts
                        let npts = npts
                            .into_iter()
                            .map(|x| norms[x.unwrap()])
                            .collect::<Box<[_]>>();
                        // collect and set
                        let nverts = Mat3::from([npts[0], npts[1], npts[2]]);
                        fragshader.set_tri_normal(Some(nverts));
                    } else {
                        // if not, unset
                        fragshader.set_tri_normal(None);
                    }

                    // raster the tri
                    rasterize(ctx.mutate(), &fragshader, tri);
                }
            }
        }
    }

    // show output
    ctx.copy_frame().save("render.tga").unwrap();

    // show zbuf
    ctx.render_zbuf().save("zbuf.tga").unwrap();
    ctx.clear_bufs();

    Ok(())
}
