#![recursion_limit = "128"]
use obj::Obj;

#[allow(dead_code)]
mod vec;
use crate::vec::*;
mod gl_func;
use crate::gl_func::*;

// https://haqr.eu/tinyrenderer
fn main() {
    let mut ctx = OwnedDrawingContext::new(1024, 1024);
    let eye = Vec3::from([-1., 0., 2.]);
    let center = Vec3::from([0., 0., 0.]);
    // this is inverted due to how coords are upside down in image
    let up = Vec3::from([0., -1., 0.]);
    let perspective = perpsective_mat((eye - center).norm());
    let modelview = modelview_mat(eye, center, up);

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
        let pt0 = Vec3::from(vert0).extend(1.);
        let pt1 = Vec3::from(vert1).extend(1.);
        let pt2 = Vec3::from(vert2).extend(1.);
        // make clip(triangle) coords
        let tri = Mat3x4::from([
            perspective * modelview * pt0,
            perspective * modelview * pt1,
            perspective * modelview * pt2,
        ]);

        rasterize(ctx.mutate(), tri);
    }

    // show output
    ctx.copy_frame().save("arf.tga").unwrap();

    // show zbuf
    ctx.render_zbuf().save("zbuf.tga").unwrap();
    ctx.clear_bufs();
}
