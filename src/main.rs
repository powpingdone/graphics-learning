#![recursion_limit = "128"]
use std::ffi::OsStr;
use std::fmt::Pointer;
use std::fs::File;
use std::io::BufReader;

use image::ImageReader;
use obj::Obj;

#[allow(dead_code)]
mod vec;
use crate::vec::*;
mod gl_func;
use crate::gl_func::*;

// lighting prepass: compute zbuf on model to see where the light can reach
struct LightingCompute {
    perspective: Mat4,
    modelview: Mat4,
}

impl LightingCompute {
    fn new(sun: Vec3, center: Vec3, up: Vec3) -> Self {
        Self {
            perspective: perpsective_mat((sun - center).norm()),
            modelview: modelview_mat(sun, center, up),
        }
    }

    fn vertex_transform(&mut self, pt0: Vec3, pt1: Vec3, pt2: Vec3) -> Mat3x4 {
        // warp each vertex through the model viewpoint and perspective
        Mat3x4::from([
            self.perspective * self.modelview * pt0.extend(1.),
            self.perspective * self.modelview * pt1.extend(1.),
            self.perspective * self.modelview * pt2.extend(1.),
        ])
    }

    fn show_transform_mats(&self) -> (Mat4, Mat4) {
        (self.perspective, self.modelview)
    }
}

impl FragShader for LightingCompute {
    // the reason that this is simple is that all we want is the zbuf for a triangle
    fn fragment(&self, _: Vec3) -> Option<Vec4> {
        Some(Vec4::default())
    }
}

// texture drawing
struct ModelDrawFragShader {
    perspective: Mat4,
    modelview: Mat4,
    tri: Mat3,
    sun: Vec3,
    sun_zbuf: Vec<f32>,
    sun_zbuf_wh: (usize, usize),
    sun_transform: Mat4,
    frag_transform: Mat4,
    tri_normal: Option<Mat3>,
    nm_image: Option<image::RgbaImage>,
    diffuse_image: Option<image::RgbaImage>,
    specular_image: Option<image::GrayImage>,
    nm_tan_image: Option<image::RgbaImage>,
    tri_texture: Option<Mat3x2>,
}

impl ModelDrawFragShader {
    fn new(
        eye: Vec3,
        center: Vec3,
        up: Vec3,
        viewport: Mat4,
        sun: Vec3,
        sun_zbuf: Vec<f32>,
        sun_zbuf_wh: (usize, usize),
        sun_perspective: Mat4,
        sun_modelview: Mat4,
        sun_viewport: Mat4,
    ) -> Self {
        let perspective = perpsective_mat((eye - center).norm());
        let modelview = modelview_mat(eye, center, up);
        Self {
            perspective,
            modelview,
            tri: Mat3::default(),
            tri_normal: None,
            nm_image: None,
            nm_tan_image: None,
            tri_texture: None,
            diffuse_image: None,
            specular_image: None,
            sun,
            sun_zbuf,
            sun_zbuf_wh,
            sun_transform: sun_perspective * sun_modelview * sun_viewport,
            frag_transform: (perspective * modelview * viewport).invert(),
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

    fn set_nm_image(&mut self, inp: Option<image::RgbaImage>) {
        self.nm_image = inp;
    }

    fn set_nm_tangent_image(&mut self, inp: Option<image::RgbaImage>) {
        self.nm_tan_image = inp;
    }

    fn set_diffuse_image(&mut self, inp: Option<image::RgbaImage>) {
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
fn get_reflected_uv_coord(frag_coord: Vec3, inp_coords: Mat3x2, dims: (u32, u32)) -> (u32, u32) {
    // find coords in the triangle
    let coord: Vec2 = map_frag_coord_and_matrix(frag_coord, inp_coords.into_iter());

    // convert coords to px indexes
    let x = (dims.0 as f32 * coord[X]).round().max(1.) as u32;
    let y = (dims.1 as f32 * coord[Y]).round().max(1.) as u32;
    (x - 1, y - 1)
}

impl FragShader for ModelDrawFragShader {
    fn fragment(&self, frag_coord: Vec3) -> Option<Vec4> {
        // base color to be manipulated
        let model_color = if let Some(mat) = self.tri_texture
            && let Some(ref img) = self.diffuse_image
        {
            // pull from the diffuse image
            let (x, y) = get_reflected_uv_coord(frag_coord, mat, img.dimensions());
            let px = img
                .get_pixel(x, y)
                .0
                .into_iter()
                .map(|x| x as f32)
                .collect::<Box<[_]>>();
            Vec4::from([px[0], px[1], px[2], px[3]])
        } else {
            // set as a "notexture", which is alternating purple and black
            if frag_coord.into_iter().sum::<f32>() % 1. > 0.5 {
                Vec4::from([255., 0., 255., 255.])
            } else {
                Vec4::from([0., 0., 0., 255.])
            }
        };

        // the percentage of ambient lighting, the base of the model color
        let ambient = 0.5;

        // the texturing of the surface in relation to lighting
        let surface_norm = {
            if let Some(uv_mat) = self.tri_texture
                && let Some(ref img) = self.nm_tan_image
                && let Some(norm_mat) = self.tri_normal
            {
                // use precomputed tangent normal map
                let (x, y) = get_reflected_uv_coord(frag_coord, uv_mat, img.dimensions());
                // fetch pixel and convert to [-1, 1]
                let px = img
                    .get_pixel(x, y)
                    .0
                    .to_owned()
                    .into_iter()
                    .map(|x| 2. * (x as f32 / 255.) - 1.)
                    .collect::<Box<[_]>>();
                let px = Vec4::from([px[0], px[1], px[2], 1.]);
                // compute tangent normal offsets
                let edge0 = self.tri[1] - self.tri[0];
                let edge1 = self.tri[2] - self.tri[0];
                let edge = Mat2x4::from([edge0.extend(1.), edge1.extend(1.)]);
                let uv_edge0 = uv_mat[1] - uv_mat[0];
                let uv_edge1 = uv_mat[2] - uv_mat[0];
                let uv_edge = Mat2::from([uv_edge0, uv_edge1]);

                // compute tangent
                let tangent_bitangent = edge * uv_edge.invert();
                let basis_tangent = Mat4::from([
                    tangent_bitangent[0].normalize(),
                    tangent_bitangent[1].normalize(),
                    norm_mat
                        .iter()
                        .copied()
                        .zip(frag_coord.iter().copied())
                        .map(|(e, i)| e.extend(1.) * i)
                        .sum::<Vec4>()
                        .normalize(),
                    [0., 0., 0., 1.].into(),
                ]);

                // finally get norm coord
                let fin = basis_tangent.t() * px;
                (fin.shrink() / fin[W]).normalize()
            } else if let Some(mat) = self.tri_texture
                && let Some(ref img) = self.nm_image
            {
                // use precomputed normal map
                let (x, y) = get_reflected_uv_coord(frag_coord, mat, img.dimensions());
                // fetch pixel and convert to [-1, 1]
                let px = img
                    .get_pixel(x, y)
                    .0
                    .to_owned()
                    .into_iter()
                    .map(|x| 2. * (x as f32 / 255.) - 1.)
                    .collect::<Box<[_]>>();
                // put into vec4
                Vec3::from([px[0], px[1], px[2]])
            } else if let Some(mat) = self.tri_normal {
                // compute surface normal if we have it and the normal map doesn't exist
                // vary the normals based on how far away the normal is from the bary coord
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

        // compute shadow coord
        let shadow = self.sun_transform * self.frag_transform * frag_coord.extend(1.);
        let shadow = shadow.shrink() / shadow[W];
        // then check if we're in the sun's shadow
        let x = ((self.sun_zbuf_wh.0 as f32 * shadow[X]).round()).max(0.) as usize;
        let y = ((self.sun_zbuf_wh.1 as f32 * shadow[Y]).round()).max(0.) as usize;
        let z = shadow[Z];
        let shadow_factor = if (self.sun_zbuf[x + y * self.sun_zbuf_wh.1] - z).abs() > 1e-4 {
            // we are in shadow
            0.5
        } else {
            // we are in light
            1.
        };

        // then see, in the z direction, how reflective the surface should be
        let specular = reflect[Z].max(0.).powf({
            if let Some(mat) = self.tri_texture
                && let Some(ref img) = self.specular_image
            {
                // use the specular map
                let (x, y) = get_reflected_uv_coord(frag_coord, mat, img.dimensions());
                let px = img.get_pixel(x, y).0;
                px[0] as f32
            } else {
                // default to "basically" no reflection
                0.
            }
        });

        // finally, the actual combining part to realize the pixel
        let final_model_color = model_color
            .swiz(R | G | B)
            .apply(|i| i * (ambient + 0.5 * diffuse + 0.8 * specular).min(1.) * shadow_factor)
            .extend(model_color[A]);
        Some(final_model_color)
    }
}

fn load_img(path: &OsStr, postfix: &str, text_ext: &OsStr) -> Option<image::RgbaImage> {
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
        .into_rgba8()
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
    let eye = Vec3::from([3., 0., 6.]);
    let center = Vec3::from([0., 0., 0.]);
    // this is inverted due to how coords are upside down in image
    let up = Vec3::from([0., -1., 0.]);
    let sun = Vec3::from([-1., 0., 1.]);

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

    // load objects
    let objects = args
        .chunks_exact(2)
        .map(|i| {
            let path = i[0].to_owned();
            let ext = i[1].to_owned();
            println!("loading object {}", path.display());
            (Obj::load(&path).unwrap().data, path, ext)
        })
        .collect::<Box<[_]>>();

    // render shadow zbuf
    let mut shadow_ctx = OwnedDrawingContext::new(1024, 1024);
    let mut shadow_zbuf = shadow_ctx.make_zbuf();
    let mut blank_shader = LightingCompute::new(sun, center, up);
    for (obj, _, _) in objects.iter() {
        let verts = &obj.position;
        // per object
        for object in &obj.objects {
            println!("shadowing object {}", object.name);
            // per group of faces
            for group in &object.groups {
                let faces = &group.polys;
                println!(
                    "shadowing object group \"{}\" with face count {}",
                    group.name,
                    faces.len()
                );

                // per polygon
                for face in faces.iter() {
                    // decode and pack
                    let pts = [face.0[0].0, face.0[1].0, face.0[2].0];
                    let verts = pts
                        .into_iter()
                        .map(|x| Vec3::from(verts[x]))
                        .collect::<Box<[_]>>();
                    // transform
                    let tri = blank_shader.vertex_transform(verts[0], verts[1], verts[2]);
                    // render
                    rasterize(shadow_ctx.mutate(), &blank_shader, tri, &mut shadow_zbuf);
                }
            }
        }
    }
    let shadow_zbuf = shadow_zbuf;
    let (shadow_persp, shadow_mv) = blank_shader.show_transform_mats();
    let shadow_viewport = shadow_ctx.show_viewport();
    shadow_ctx
        .render_zbuf(&shadow_zbuf)
        .save("light_zbuf.tga")
        .unwrap();
    drop((shadow_ctx, blank_shader));

    // render image
    let mut ctx = OwnedDrawingContext::new(2048, 2048);
    let mut zbuf = ctx.make_zbuf();
    let mut main_shader = ModelDrawFragShader::new(
        eye,
        center,
        up,
        ctx.show_viewport(),
        sun,
        shadow_zbuf,
        (1024, 1024),
        shadow_persp,
        shadow_mv,
        shadow_viewport,
    );
    for (obj, path, text_ext) in objects.iter() {
        let verts = &obj.position;
        let norms = &obj.normal;
        let uv = &obj.texture;

        // load textures
        main_shader.set_nm_image(load_img(path, "nm", text_ext));
        main_shader.set_nm_tangent_image(load_img(path, "nm_tangent", text_ext));
        main_shader.set_diffuse_image(load_img(path, "diffuse", text_ext));
        main_shader.set_specular_image(
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
                    main_shader.set_tri_from_coords(pt0, pt1, pt2);
                    let tri = main_shader.copy_tri_arr();

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
                        main_shader.set_tri_uv(Some(uv_verts));
                    } else {
                        // unset otherwise
                        main_shader.set_tri_uv(None);
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
                        main_shader.set_tri_normal(Some(nverts));
                    } else {
                        // if not, unset
                        main_shader.set_tri_normal(None);
                    }

                    // raster the tri
                    rasterize(ctx.mutate(), &main_shader, tri, &mut zbuf);
                }
            }
        }
    }

    // show output
    ctx.copy_frame().save("render.tga").unwrap();

    // show zbuf
    ctx.render_zbuf(&zbuf).save("zbuf.tga").unwrap();
    ctx.clear_bufs();

    Ok(())
}
