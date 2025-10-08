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
    ($name:ident => $size:expr) => {
        // Construct this struct via $name::from or $name::default
        pub struct $name<T>([T; $size]);

        impl<T> $name<T> {
            const LEN: usize = $size;
            
            const fn len(&self) -> usize {
                Self::LEN
            }
        }

        // dot product
        impl<T: Default + Copy + Add<Output = T> + Mul<Output = T>> $name<T> {
            pub fn dot(&self, rhs: $name<T>) -> T {
                self.0
                    .iter()
                    .zip(rhs.0.iter())
                    .fold(Default::default(), |acc, x| acc + *x.0 * *x.1)
            }
        }

        // common traits
        impl<T: Default> Default for $name<T> {
            fn default() -> Self {
                Self(Default::default())
            }
        }

        impl<T: Clone> Clone for $name<T> {
            fn clone(&self) -> Self {
                Self(self.0.clone())
            }
        }

        impl<T: Copy> Copy for $name<T> {}

        impl<T> From<[T; $size]> for $name<T> {
            fn from(i: [T; $size]) -> Self {
                Self(i)
            }
        }

        // indexing
        impl<T> Index<usize> for $name<T> {
            type Output = T;

            fn index(&self, index: usize) -> &T {
                &self.0[index]
            }
        }

        impl<T> IndexMut<usize> for $name<T> {
            fn index_mut(&mut self, index: usize) -> &mut T {
                &mut self.0[index]
            }
        }

        // glsl defined operations: add, sub, mul, and div are component wise
        impl<T: Add<Output = T> + Copy> Add for $name<T> {
            type Output = $name<T>;

            fn add(mut self, rhs: $name<T>) -> $name<T> {
                for x in 0..Self::LEN {
                    self[x] = self[x] + rhs[x];
                }
                self
            }
        }

        impl<T: Sub<Output = T> + Copy> Sub for $name<T> {
            type Output = $name<T>;

            fn sub(mut self, rhs: $name<T>) -> $name<T> {
                for x in 0..Self::LEN {
                    self[x] = self[x] - rhs[x];
                }
                self
            }
        }
        
        impl<T: Mul<Output = T> + Copy> Mul for $name<T> {
            type Output = $name<T>;

            fn mul(mut self, rhs: $name<T>) -> $name<T> {
                for x in 0..Self::LEN {
                    self[x] = self[x] * rhs[x];
                }
                self
            }
        }

        impl<T: Div<Output = T> + Copy> Div for $name<T> {
            type Output = $name<T>;

            fn div(mut self, rhs: $name<T>) -> $name<T> {
                for x in 0..Self::LEN {
                    self[x] = self[x] / rhs[x];
                }
                self
            }
        }

        // broadcast ops, also glsl defined.
        impl<T: Add<Output = T> + Copy> Add<T> for $name<T> {
            type Output = $name<T>;

            fn add(mut self, rhs: T) -> $name<T> {
                for x in 0..Self::LEN {
                    self[x] = self[x] + rhs;
                }
                self
            }
        }

        impl<T: Sub<Output = T> + Copy> Sub<T> for $name<T> {
            type Output = $name<T>;

            fn sub(mut self, rhs: T) -> $name<T> {
                for x in 0..Self::LEN {
                    self[x] = self[x] - rhs;
                }
                self
            }
        }
        
        impl<T: Mul<Output = T> + Copy> Mul<T> for $name<T> {
            type Output = $name<T>;

            fn mul(mut self, rhs: T) -> $name<T> {
                for x in 0..Self::LEN {
                    self[x] = self[x] * rhs;
                }
                self
            }
        }
        
        impl<T: Div<Output = T> + Copy> Div<T> for $name<T> {
            type Output = $name<T>;

            fn div(mut self, rhs: T) -> $name<T> {
                for x in 0..Self::LEN {
                    self[x] = self[x] / rhs;
                }
                self
            }
        }
    };
}

vec_gen!(Vec2 => 2);
vec_gen!(Vec3 => 3);
vec_gen!(Vec4 => 4);

// mat

// transpose
// let mut ret = Mat::new();
// for x in 0..X::USIZE {
//     for y in 0..Y::USIZE {
//         ret[y][x] = self[x][y];
//     }
// }
// ret

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

// indexing

// standard ops aka add and sub

// broadcast scalar

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
