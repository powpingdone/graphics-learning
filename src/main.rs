#![recursion_limit = "128"]
use obj::Obj;

#[allow(dead_code)]
mod vec;
use crate::vec::*;
mod gl_func;
use crate::gl_func::*;

struct PhongFragShaderWithModel {
    perspective: Mat4,
    modelview: Mat4,
    tri: Mat3,
    sun: Vec3,
}

impl PhongFragShaderWithModel {
    fn new(eye: Vec3, center: Vec3, up: Vec3, sun: Vec3) -> Self {
        Self {
            perspective: perpsective_mat((eye - center).norm()),
            modelview: modelview_mat(eye, center, up),
            tri: Mat3::default(),
            sun,
        }
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

    fn copy_tri_arr(&self) -> Mat3x4 {
        // extract out the homogenous triangle struct for the draw call
        self.tri.apply(|row| row.extend(1.)).into()
    }
}

impl FragShader for PhongFragShaderWithModel {
    fn fragment(&self, _frag_coord: Vec3) -> Option<Vec3> {
        // base color to be manipulated
        let model_color = Vec3::from([255., 255., 255.]);
        // the percentage of ambient lighting, the base of the model color
        let ambient = 0.4;
        // get AB and AC vectors
        let a = self.tri[1] - self.tri[0];
        let b = self.tri[2] - self.tri[0];
        // to cross and make the normal vector
        let surface_norm = a.cross(b).normalize();
        // and then find how far away it is from the sun
        let diffuse = surface_norm.dot(self.sun).max(0.);
        // then, make opposite of the sun vector for reflection
        let r = (surface_norm * surface_norm.dot(self.sun) * 2. - self.sun).normalize();
        // then see, in the z direction, how reflective the surface should be
        let specular = r[Z].max(0.).powi(36);
        // finally, the actual combining part
        let final_model_color =
            model_color.apply(|i| i * (ambient + 0.5 * diffuse + 0.5 * specular).min(1.));
        Some(final_model_color)
    }
}

// https://haqr.eu/tinyrenderer
fn main() {
    let mut ctx = OwnedDrawingContext::new(1024, 1024);
    let eye = Vec3::from([-1., 0., 2.]);
    let center = Vec3::from([0., 0., 0.]);
    // this is inverted due to how coords are upside down in image
    let up = Vec3::from([0., -1., 0.]);
    let sun = Vec3::from([1., 0., 1.]).normalize();
    let mut fragshader = PhongFragShaderWithModel::new(eye, center, up, sun);

    // extract out arglist
    let mut args = std::env::args_os();
    drop(args.next()); // skip executable name
    let args = args.collect::<Vec<_>>();
    if args.len() == 0 {
        println!("no args were provided, please provide paths to each .obj file");
        return;
    }

    // per file path
    for path in args {
        let obj = Obj::load(path).unwrap().data;
        let verts = &obj.position;
        // per object
        for object in &obj.objects {
            // per group of faces
            for group in &object.groups {
                let faces = &group.polys;

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
}
