// Vect: a vector of f32
// Mat: a matrix of f32
use generic_array::*;
use std::ops::*;
use typenum::*;

#[repr(transparent)]
#[derive(Clone)]
pub struct Vect<I>
where
    I: ArrayLength,
{
    vec: GenericArray<f32, I>,
}

impl<I> Vect<I>
where
    I: ArrayLength,
{
    pub fn new<const N: usize>(vec: [f32; N]) -> Self
where
Const<N>: ToUInt,
U<N>: ArrayLength,
 {
        const {
            assert!(N > 0);
        }
        Self { vec: GenericArray::from_iter(vec) }
    }

    // vectors have dot product
    pub fn dot(&self, rhs: Vect<I>) -> f32 {
        self.vec
            .iter()
            .zip(rhs.vec.iter())
            .map(|(lhs, rhs)| lhs * rhs)
            .sum()
    }
}

impl<const I: usize> Default for Vect<U<I>>
where
    Const<I>: ToUInt,
    U<I>: ArrayLength
{
    fn default() -> Self {
        Self::new([0.0; I])
    }
}

impl<const I: usize> Index<usize> for Vect<U<I>>
where
    Const<I>: ToUInt,
    U<I>: ArrayLength,
{
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        &self.vec[index]
    }
}

impl<const I: usize> IndexMut<usize> for Vect<I>
where
    Const<I>: ToUInt,
    U<I>: ArrayLength,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.vec[index]
    }
}

#[repr(transparent)]
#[derive(Clone)]
pub struct Mat<X, Y>
where
    X: ArrayLength,
    Y: ArrayLength,
{
    mat: GenericArray<Vect<Y>, X>,
}

impl<const X: usize, const Y: usize> Mat<X, Y>
where
    Const<X>: ToUInt,
    U<X>: ArrayLength,
    Const<Y>: ToUInt,
    U<Y>: ArrayLength,
{
    pub fn new(n_mat: [[f32; Y]; X]) -> Self {
        const {
            assert!(X > 0);
            assert!(Y > 0);
        }

        let mut mat: [Vect<Y>; X] = std::array::from_fn(|_| Vect::<Y>::new([0.; Y]));
        for i in 0..X {
            mat[i] = Vect::new(n_mat[i]);
        }

        Self { mat: mat.into() }
    }

    pub fn t(&self) -> Mat<Y, X> {
        let mut ret = Mat::default();
        for x in 0..X {
            for y in 0..Y {
                ret[y][x] = self[x][y];
            }
        }
        ret
    }
}

impl<const I: usize> Mat<I, I>
where
    Const<I>: ToUInt,
    U<I>: ArrayLength,
{
    fn cofactor(&self, x: usize, y: usize) -> f32 {
        let mut submat = Mat::<{ U<X> as Sub<P1> }, { U<Y> as Sub<P1> }>::default();
        for i in 0..(I - 1) {
            for j in 0..(I - 1) {
                submat[i][j] = self[i + ((i >= x) as usize)][j + ((j >= y) as usize)]
            }
        }
        submat.det() * (if (x + y) % 2 == 1 { -1. } else { 1. })
    }

    fn det(&self) -> f32 {
        if const { I == 1 } {
            // recurse end
            self[0][0]
        } else {
            // make sum of cofactors
            (0..I).fold(0.0, |acc, i| acc + self[0][i] * self.cofactor(0, i))
        }
    }

    pub fn invert_t(&self) -> Self {
        let mut adjugate_t = Mat::<I, I>::default();
        for i in 0..(I - 1) {
            for j in 0..(I - 1) {
                adjugate_t[i][j] = self.cofactor(i, j);
            }
        }

        adjugate_t.clone() / (adjugate_t[0].dot(self[0]))
    }

    pub fn invert(&self) -> Self {
        self.invert_t().t()
    }
}

impl<const X: usize, const Y: usize> Default for Mat<X, Y>
where
    Const<X>: ToUInt,
    U<X>: ArrayLength,
    Const<Y>: ToUInt,
    U<Y>: ArrayLength,
{
    // this is defined here so that the generics check at Self::new are activated
    fn default() -> Self {
        Self::new([[0.0; Y]; X])
    }
}

impl<const X: usize, const Y: usize> Index<usize> for Mat<X, Y>
where
    Const<X>: ToUInt,
    U<X>: ArrayLength,
    Const<Y>: ToUInt,
    U<Y>: ArrayLength,
{
    type Output = Vect<Y>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.mat[index]
    }
}

impl<const X: usize, const Y: usize> IndexMut<usize> for Mat<X, Y>
where
    Const<X>: ToUInt,
    U<X>: ArrayLength,
    Const<Y>: ToUInt,
    U<Y>: ArrayLength,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.mat[index]
    }
}

//impl<const X: usize, const Y: usize> Add<f32> for Mat<X, Y> {
//    type Output = Self;
//
//    fn add(self, rhs: f32) -> Self::Output {
//        let mut new = self.clone();
//        for x in 0..X {
//            for y in 0..Y {
//                new[x][y] += rhs;
//            }
//        }
//        new
//    }
//}
//
//impl<const X: usize, const Y: usize> Add<Mat<X, Y>> for f32 {
//    type Output = Mat<X, Y>;
//
//    fn add(self, rhs: Mat<X, Y>) -> Self::Output {
//        let mut new = rhs.clone();
//        for x in 0..X {
//            for y in 0..Y {
//                new[x][y] += self;
//            }
//        }
//        new
//    }
//}
//
//impl<const X: usize, const Y: usize> Sub<Mat<X, Y>> for f32 {
//    type Output = Mat<X, Y>;
//
//    fn sub(self, rhs: Mat<X, Y>) -> Self::Output {
//        let mut new = rhs.clone();
//        for x in 0..X {
//            for y in 0..Y {
//                new[x][y] -= self;
//            }
//        }
//        new
//    }
//}
//
//impl<const X: usize, const Y: usize> Sub<f32> for Mat<X, Y> {
//    type Output = Self;
//
//    fn sub(self, rhs: f32) -> Self::Output {
//        let mut new = self.clone();
//        for x in 0..X {
//            for y in 0..Y {
//                new[x][y] -= rhs;
//            }
//        }
//        new
//    }
//}
//
//impl<const X: usize, const Y: usize> Mul<Mat<X, Y>> for f32 {
//    type Output = Mat<X, Y>;
//
//    fn mul(self, rhs: Mat<X, Y>) -> Self::Output {
//        let mut new = rhs.clone();
//        for x in 0..X {
//            for y in 0..Y {
//                new[x][y] *= self;
//            }
//        }
//        new
//    }
//}
//
//impl<const X: usize, const Y: usize> Mul<f32> for Mat<X, Y> {
//    type Output = Self;
//
//    fn mul(self, rhs: f32) -> Self::Output {
//        let mut new = self.clone();
//        for x in 0..X {
//            for y in 0..Y {
//                new[x][y] *= rhs;
//            }
//        }
//        new
//    }
//}
//
//impl<const X: usize, const Y: usize> Div<f32> for Mat<X, Y> {
//    type Output = Self;
//
//    fn div(self, rhs: f32) -> Self::Output {
//        let mut new = self.clone();
//        for x in 0..X {
//            for y in 0..Y {
//                new[x][y] /= rhs;
//            }
//        }
//        new
//    }
//}
//
//impl<const X: usize, const Y: usize, const Y2: usize> Mul<Mat<Y, Y2>> for Mat<X, Y> {
//    type Output = Mat<X, Y2>;
//    // [[f32; Y]; X] * [[f32; Y2]; Y] = [[f32; Y2]; X]
//    // self[X][Y] * rhs[Y][Y2] = new[X][Y2]
//    fn mul(self, rhs: Mat<Y, Y2>) -> Self::Output {
//        let mut new = Mat::default();
//        for x in 0..X {
//            for y in 0..Y2 {
//                for i in 0..Y {
//                    new[x][y] += self[x][i] * rhs[i][y];
//                }
//            }
//        }
//        new
//    }
//}
