// Vect: a vector of f32
// Mat: a matrix of f32
use std::ops::*;

// access enums
// basically: typestate spam via macros to index into vecs/mats with coords
// How it would work is you have 3 seperate coords systems: XYZW, RGBA, and STPQ
// where the first letter corresponds to [0], second corresponds to [1], and so on.
// You can apply a bitwise or (X | Y) to each and get a combined access aka swizzling.
// But, there is an issue: compile time checking and repeating characters (aka: XXYY).
// The former can be fixed by only allowing certain typestates be indexable to certain vecs.
// IE: XYZ can only index into xVec3 and xVec4, but not xVec2 due to size.
// For such a thing to be done, each bitwise or will produce a new "combined" type.
// Example:
// (X | Y) => Vec2Access(X, Y)
// (X | Z | Z) => (Vec2Access(X, Z) | Z) => Vec3Access(X, Z, Z)
// These then return Vecs of their own, based on their size. This also conviently fixes the issue of
// repeating characters, as each field of the struct is set to be that character.
//
// TODO

macro_rules! vec_gen {
    ($name:ident => $size:expr, $type:ty) => {
        // Construct this struct via $name::from or $name::default
        #[derive(Debug, Default, Clone, Copy, PartialEq)]
        pub struct $name([$type; $size]);

        impl $name {
            pub const LEN: usize = $size;

            pub const fn len(&self) -> usize {
                Self::LEN
            }
        }

        // dot product
        impl $name {
            pub fn dot(&self, rhs: $name) -> $type {
                self.0
                    .iter()
                    .zip(rhs.0.iter())
                    .fold(Default::default(), |acc, x| acc + *x.0 * *x.1)
            }
        }

        // common traits
        impl From<[$type; $size]> for $name {
            fn from(i: [$type; $size]) -> Self {
                Self(i)
            }
        }

        // indexing
        impl Index<usize> for $name {
            type Output = $type;

            fn index(&self, index: usize) -> &$type {
                &self.0[index]
            }
        }

        impl IndexMut<usize> for $name {
            fn index_mut(&mut self, index: usize) -> &mut $type {
                &mut self.0[index]
            }
        }

        // normal broadcast ops
        vec_broadcast_op!($name ($type) => Add (add), +);
        vec_broadcast_op!($name ($type) => Sub (sub), -);
        vec_broadcast_op!($name ($type) => Mul (mul), *);
        vec_broadcast_op!($name ($type) => Div (div), /);
        vec_broadcast_op!($name ($type) => Rem (rem), %);


        // int specific broadcast ops
        //vec_broadcast_op!($name => Shl (shl), <<, $type);
        //vec_broadcast_op!($name => Shr (shr), >>, $type);
        //vec_broadcast_op!($name => BitAnd (bitand), &, $type);
        //vec_broadcast_op!($name => BitOr (bitor), |, $type);
        //vec_broadcast_op!($name => BitXor (bitxor), ^, $type);
    };
}

// glsl ops are broadcast for any vec2vec or vec2scalar operation
macro_rules! vec_broadcast_op {
    ($name:ident ($type:ty) => $trait_name:ident ($trait_fn:ident), $op_by:tt) => {
        vec_broadcast_indivdual_impl!($name ($type) => $trait_name ($trait_fn), $op_by);
        vec_broadcast_indivdual_impl!($name => $trait_name ($trait_fn), $op_by);
    };
}

macro_rules! vec_broadcast_indivdual_impl {
    ($name:ident ($type:ty) => $trait_name:ident ($trait_fn:ident), $op_by:tt) => {
        impl $trait_name<$type> for $name {
            type Output = $name;

            fn $trait_fn(mut self, rhs: $type) -> $name {
                for x in 0..Self::LEN {
                    self[x] = self[x] $op_by rhs;
                }
                self
            }
        }
    };

    ($name:ident => $trait_name:ident ($trait_fn:ident), $op_by:tt) => {
        impl $trait_name<$name> for $name {
            type Output = $name;

            fn $trait_fn(mut self, rhs: Self) -> $name {
                for x in 0..Self::LEN {
                    self[x] = self[x] $op_by rhs[x];
                }
                self
            }
        }
    };
}

vec_gen!(Vec2 => 2, f32);
vec_gen!(Vec3 => 3, f32);
vec_gen!(Vec4 => 4, f32);

// mat
macro_rules! mat_gen {
    ($name:ident => $vec_row:ident * $col_len:expr) => {
        // Construct this struct via $name::from or $name::default
        #[derive(Debug, Default, Clone, Copy, PartialEq)]
        pub struct $name([$vec_row; $col_len]);

        impl $name {
            pub const COLS: usize = $col_len;
            pub const ROWS: usize = $vec_row::LEN;

            pub const fn cols(&self) -> usize {
                Self::COLS
            }

            pub const fn rows(&self) -> usize {
                Self::ROWS
            }
        }

        // common traits
        impl From<[$vec_row; $col_len]> for $name {
            fn from(i: [$vec_row; $col_len]) -> Self {
                Self(i)
            }
        }

        // indexing
        impl Index<usize> for $name {
            type Output = $vec_row;

            fn index(&self, index: usize) -> &$vec_row {
                &self.0[index]
            }
        }

        impl IndexMut<usize> for $name {
            fn index_mut(&mut self, index: usize) -> &mut $vec_row {
                &mut self.0[index]
            }
        }

        // glsl scalar broadcast ops

        // glsl vector broadcast ops (EXCEPT MUL, THATS THE NEXT ONE)

        // glsl dot product vector broadcast ops
        // signatures are Vec * Mat

        // glsl vec mat matmul 
        // signatures are Mat * Vec
    };
}

mat_gen!(Mat2 => Vec2 * 2);
mat_gen!(Mat2x3 => Vec2 * 3);
mat_gen!(Mat2x4 => Vec2 * 4);
mat_gen!(Mat3x2 => Vec3 * 2);
mat_gen!(Mat3 => Vec3 * 3);
mat_gen!(Mat3x4 => Vec3 * 4);
mat_gen!(Mat4x2 => Vec4 * 2);
mat_gen!(Mat4x3 => Vec4 * 3);
mat_gen!(Mat4 => Vec4 * 4);

// transpose macro
macro_rules! mat_t {
    ($self:ident) => {
        mat_t!($self => $self);
    };

    ($l:ident => $r:ident) => {
        impl $l {
            pub fn t(self) -> $r {
                const _CHECK_ROW_COL: () = assert!($l::COLS == $r::ROWS && $l::ROWS == $r::COLS); 

                let mut new = $r::default();
                for i in 0..Self::COLS {
                    for j in 0..Self::ROWS {
                        new[j][i] = self[i][j];
                    }
                }
                new
            }
        }
    };
}

mat_t!(Mat2);
mat_t!(Mat3);
mat_t!(Mat4);

mat_t!(Mat2x3 => Mat3x2);
mat_t!(Mat2x4 => Mat4x2);
mat_t!(Mat3x2 => Mat2x3);
mat_t!(Mat3x4 => Mat4x3);
mat_t!(Mat4x2 => Mat2x4);
mat_t!(Mat4x3 => Mat3x4);

// invert_t
// let mut adjugate_t = Mat::default();
// for i in 0..(I::USIZE - 1) {
//     for j in 0..(I::USIZE - 1) {
//         adjugate_t[i][j] = self.cofactor(i, j);
//     }
// }
// adjugate_t.clone() / (adjugate_t[0].dot(&self[0]))

// cofactor
// let mut submat = Mat::<<I as Sub<U1>>::Output, <I as Sub<U1>>::Output>::new();
// for i in 0..(I::USIZE - 1) {
//     for j in 0..(I::USIZE - 1) {
//         submat[i][j] = self[i + ((i >= x) as usize)][j + ((j >= y) as usize)];
//     }
// }
// submat.det() * (if (x + y) % 2 == 1 { -1. } else { 1. })

// det
// if const { I::USIZE == 1 } {
//     self[0][0]
// } else {
//     (0..I::USIZE).fold(0.0, |acc, i| acc + self[0][i] * self.cofactor(0, i))
// }

// standard ops aka add and sub


// broadcast vector

// matmul
// let mut ret = Mat::new();
// // [[f32; Y]; X] * [[f32; Y2]; Y] = [[f32; Y2]; X]
// // self[X][Y] * rhs[Y][Y2] = new[X][Y2]
// for x in 0..X::USIZE {
//     for y in 0..Y::USIZE {
//         for i in 0..M::USIZE {
//             ret[x][y] += self[x][i] * rhs[i][y];
//         }
//     }
// }
// ret
