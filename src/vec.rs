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
    ($name:ident => $vec_row:ident ($type:ty) * $col_len:expr) => {
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
        mat_broadcast_scalar!($name ($type) => Add (add), +);
        mat_broadcast_scalar!($name ($type) => Sub (sub), -);
        mat_broadcast_scalar!($name ($type) => Mul (mul), *);
        mat_broadcast_scalar!($name ($type) => Div (div), /);
        mat_broadcast_scalar!($name ($type) => Rem (rem), %);

        // glsl dot product via vector
        // signature is Vec * Mat
        impl Mul<$name> for $vec_row {
            type Output = $vec_row;

            fn mul(self, rhs: $name) -> $vec_row {
                let mut new = $vec_row::default();
                for x in 0..$name::COLS {
                    new[x] = self.dot(rhs[x]);
                }
                new
            }
        }

        // glsl vec mat matmul
        // signature is Mat * Vec
        impl Mul<$vec_row> for $name {
            type Output = $vec_row;

            fn mul(self, rhs: $vec_row) -> $vec_row {
                let mut new = $vec_row::default();
                for row in 0..Self::ROWS {
                    for col in 0..Self::COLS {
                        new[col] += self[col][row] * rhs[col];
                    }
                }
                new
            }
        }

        // glsl mat standard broadcast ops (except matmul, tis a separate macro)
        mat_broadcast_mat!($name => Add (add), +);
        mat_broadcast_mat!($name => Sub (sub), -);
        mat_broadcast_mat!($name => Div (div), /);
        mat_broadcast_mat!($name => Rem (rem), %);
    };
}

macro_rules! mat_broadcast_scalar {
    ($name:ident ($type:ty) => $trait_name:ident ($trait_fn:ident), $op_by:tt) => {
        impl $trait_name<$type> for $name {
            type Output = $name;

            fn $trait_fn(mut self, rhs: $type) -> $name {
                for x in 0..Self::COLS {
                    for y in 0..Self::ROWS {
                        self[x][y] = self[x][y] $op_by rhs;
                    }
                }
                self
            }
        }
    };
}

macro_rules! mat_broadcast_mat {
    ($name:ident => $trait_name:ident ($trait_fn:ident), $op_by:tt) => {
        impl $trait_name<$name> for $name {
            type Output = $name;

            fn $trait_fn(mut self, rhs: $name) -> $name {
                for x in 0..Self::COLS {
                    for y in 0..Self::ROWS {
                        self[x][y] = self[x][y] $op_by rhs[x][y];
                    }
                }
                self
            }
        }
    };
}

mat_gen!(Mat2x2 => Vec2 (f32) * 2);
mat_gen!(Mat2x3 => Vec3 (f32) * 2);
mat_gen!(Mat2x4 => Vec4 (f32) * 2);
mat_gen!(Mat3x2 => Vec2 (f32) * 3);
mat_gen!(Mat3x3 => Vec3 (f32) * 3);
mat_gen!(Mat3x4 => Vec4 (f32) * 3);
mat_gen!(Mat4x2 => Vec2 (f32) * 4);
mat_gen!(Mat4x3 => Vec3 (f32) * 4);
mat_gen!(Mat4x4 => Vec4 (f32) * 4);

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

mat_t!(Mat2x2);
mat_t!(Mat3x3);
mat_t!(Mat4x4);

mat_t!(Mat2x3 => Mat3x2);
mat_t!(Mat2x4 => Mat4x2);
mat_t!(Mat3x2 => Mat2x3);
mat_t!(Mat3x4 => Mat4x3);
mat_t!(Mat4x2 => Mat2x4);
mat_t!(Mat4x3 => Mat3x4);

// matmul
macro_rules! mat_mul {
    ($lhs:ident * $rhs:ident = $out:ident) => {
        impl Mul<$rhs> for $lhs {
            type Output = $out;

            fn mul(self, rhs: $rhs) -> $out {
                const _ROW_COL_CHECK: () = assert!($lhs::COLS == $rhs::ROWS);
                const _COLS_MATCH: () = assert!($lhs::ROWS == $out::ROWS);
                const _ROWS_MATCH: () = assert!($rhs::COLS == $out::COLS);

                let mut ret = $out::default();
                for x in 0..$out::COLS {
                    for y in 0..$out::ROWS {
                        for i in 0..$lhs::COLS {
                            ret[x][y] = self[i][x] * rhs[y][i];
                        }
                    }
                }
                ret
            }
        }
    };
}

mat_mul!(Mat2x2 * Mat2x2 = Mat2x2);
mat_mul!(Mat2x2 * Mat3x2 = Mat3x2);
mat_mul!(Mat2x2 * Mat4x2 = Mat4x2);
mat_mul!(Mat2x3 * Mat2x2 = Mat2x3);
mat_mul!(Mat2x3 * Mat3x2 = Mat3x3);
mat_mul!(Mat2x3 * Mat4x2 = Mat4x3);
mat_mul!(Mat2x4 * Mat2x2 = Mat2x4);
mat_mul!(Mat2x4 * Mat3x2 = Mat3x4);
mat_mul!(Mat2x4 * Mat4x2 = Mat4x4);

mat_mul!(Mat3x2 * Mat2x3 = Mat2x2);
mat_mul!(Mat3x2 * Mat3x3 = Mat3x2);
mat_mul!(Mat3x2 * Mat4x3 = Mat4x2);
mat_mul!(Mat3x3 * Mat2x3 = Mat2x3);
mat_mul!(Mat3x3 * Mat3x3 = Mat3x3);
mat_mul!(Mat3x3 * Mat4x3 = Mat4x3);
mat_mul!(Mat3x4 * Mat2x3 = Mat2x4);
mat_mul!(Mat3x4 * Mat3x3 = Mat3x4);
mat_mul!(Mat3x4 * Mat4x3 = Mat4x4);

mat_mul!(Mat4x2 * Mat2x4 = Mat2x2);
mat_mul!(Mat4x2 * Mat3x4 = Mat3x2);
mat_mul!(Mat4x2 * Mat4x4 = Mat4x2);
mat_mul!(Mat4x3 * Mat2x4 = Mat2x3);
mat_mul!(Mat4x3 * Mat3x4 = Mat3x3);
mat_mul!(Mat4x3 * Mat4x4 = Mat4x3);
mat_mul!(Mat4x4 * Mat2x4 = Mat2x4);
mat_mul!(Mat4x4 * Mat3x4 = Mat3x4);
mat_mul!(Mat4x4 * Mat4x4 = Mat4x4);

// special functions and an alias for square mats
macro_rules! square_mat {
    ($name:ident ($alias:ident) => $typ:ident) => {
        pub type $alias = $name;

        impl $name {
            const _IS_SQUARE: () = assert!(Self::COLS == Self::ROWS);
            pub const I: usize = Self::COLS;

            pub fn identity() -> Self {
                let mut ret = Self::default();
                for x in 0..Self::COLS {
                    for y in 0..Self::ROWS {
                        ret[x][y] = 1.0;
                    }
                }
                ret
            }

            pub fn invert_t(self) -> Self {
                let mut adjugate_t = Self::default();
                for i in 0..(Self::I - 1) {
                    for j in 0..(Self::I - 1) {
                        adjugate_t[i][j] = self.cofactor(i, j);
                    }
                }
                adjugate_t.clone() / (adjugate_t[0].dot(self[0]))
            }

            pub fn invert(self) -> Self {
                self.invert_t().t()
            }

            // This is technically incorrect as a det would account for when I is 1 (Mat1 aka scalar)
            // but due to Mat2 being just above Mat1, that special case is handled there instead
            pub fn det(self) -> $typ {
                (0..Self::I).fold(0.0f32, |acc, i| acc + self[i][0] * self.cofactor(i, 0))
            }
        }
    };
}

square_mat!(Mat2x2(Mat2) => f32);
square_mat!(Mat3x3(Mat3) => f32);
square_mat!(Mat4x4(Mat4) => f32);

// then we have the inversion stuffs
macro_rules! cofactor_mat {
    ($name:ident ($typ:ident) > $lower:ident ) => {
        impl $name {
            const _IS_ONE_LOWER: () = assert!(Self::I == $lower::I + 1);

            pub fn cofactor(self, x: usize, y: usize) -> $typ {
                let mut submat = $lower::default();
                for i in 0..(Self::I - 1) {
                    for j in 0..(Self::I - 1) {
                        submat[i][j] = self[i + ((i >= x) as usize)][j + ((j >= y) as usize)];
                    }
                }
                submat.det() * (if (x + y) % 2 == 1 { -1. } else { 1. })
            }
        }
    };

    // only for Mat2, as below it is a scalar
    ($name:ident ($typ:ident)) => {
        impl $name {
            const _LOWEST_ONLY: () = assert!(Self::I - 1 == 1);

            pub fn cofactor(self, x: usize, y: usize) -> $typ {
                // Since this only leads to a scalar, manually extract it from the mat.
                // Note that cofactors *only* have everything except the selected rows, so
                // invert I over each selection and you have the scalar.
                self[Self::I - x][Self::I - y] * (if (x + y) % 2 == 1 { -1. } else { 1. })
            }
        }
    };
}

cofactor_mat!(Mat2(f32));
cofactor_mat!(Mat3(f32) > Mat2);
cofactor_mat!(Mat4(f32) > Mat3);
