// Mat: a vector/matrix of f32
use std::ops::*;

#[repr(transparent)]
#[derive(Clone, Debug)]
struct Mat<const X: usize, const Y: usize> {
    mat: [[f32; Y]; X],
}

impl<const X: usize, const Y: usize> Mat<X, Y> {
    pub const fn new(mat: [[f32; Y]; X]) -> Self {
        assert!(X != 0 && X <= 4);
        assert!(Y != 0 && Y <= 4);
        Self { mat }
    }

    // vectors have dot product, with an appropriate assert
    pub fn dot(&self, rhs: Mat<Y, X>) -> f32 {
        const {
            assert!(
                X == 1 || Y == 1,
                "expected a dim to be 1, but X or Y are not 1"
            )
        };
        let mut res = 0.0f32;
        if X == 1 {
            for i in 0..Y {
                res += self[0][i] * rhs[i][0];
            }
        } else
        /* Y == 1 */
        {
            for i in 0..X {
                res += self[i][0] * rhs[0][i];
            }
        }
        res
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

impl<const I: usize> Mat<I, I> {
    pub fn inv(&self) -> Option<Mat<I, I>> {
        if const { I == 1 } {
            let mut new = self.clone();
            new[0][0] = 1. / new[0][0];
            return Some(new);
        } else if const { I == 2 } {
            // the simple case of a 2x2 matrix having a determinant that is 1 / (ad - bc)
            let mut new = Mat::<I, I>::default();
            new[0][0] = self[1][1];
            new[1][1] = self[0][0];
            new[0][1] = -self[0][1];
            new[1][0] = -self[1][0];
            let det = det(self[0][0], self[0][1], self[1][0], self[1][1])?;
            return Some(det * new);
        } else if const { I == 3 } {
            // 3x3 inverse
            let mut cofactor = Mat::<I, I>::default();
            let mut inv = false;
            // determine cofactor
            for x in 0..3 {
                for y in 0..3 {
                    let fx = (x + 1) % 3; // first x
                    let sx = (x + 2) % 3; // second x
                    let fy = (y + 1) % 3; // first y
                    let sy = (y + 2) % 3; // second y
                    cofactor[x][y] = det(self[fx][fy], self[fx][sy], self[sx][fy], self[sx][sy])?;
                    if inv {
                        cofactor[x][y] *= -1.;
                    }
                    inv = !inv;
                }
            }
            let adjoint = cofactor.t();
            // get self det
            let self_det = self[0][0] * cofactor[0][0] - self[0][1] * cofactor[0][1]
                + self[0][2] * cofactor[0][2];
            if self_det == 0.0 {
                return None;
            }
            // return inv
            return Some(adjoint / self_det);
        } else if const { I == 4 } {
        }
        unreachable!();
    }
}

// 2x2 det function
fn det(a: f32, b: f32, c: f32, d: f32) -> Option<f32> {
    let det = a * d - b * c;
    if det == 0.0 {
        return None;
    } else {
        Some(1. / det)
    }
}

impl<const X: usize, const Y: usize> Default for Mat<X, Y> {
    // this is defined here so that the generics check at Self::new is activated
    fn default() -> Self {
        Self::new([[0.0; Y]; X])
    }
}

impl<const X: usize, const Y: usize> Index<usize> for Mat<X, Y> {
    type Output = [f32; Y];

    fn index(&self, index: usize) -> &Self::Output {
        &self.mat[index]
    }
}

impl<const X: usize, const Y: usize> IndexMut<usize> for Mat<X, Y> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.mat[index]
    }
}

impl<const X: usize, const Y: usize> Add<f32> for Mat<X, Y> {
    type Output = Self;

    fn add(self, rhs: f32) -> Self::Output {
        let mut new = self.clone();
        for x in 0..X {
            for y in 0..Y {
                new[x][y] += rhs;
            }
        }
        new
    }
}

impl<const X: usize, const Y: usize> Add<Mat<X, Y>> for f32 {
    type Output = Mat<X, Y>;

    fn add(self, rhs: Mat<X, Y>) -> Self::Output {
        let mut new = rhs.clone();
        for x in 0..X {
            for y in 0..Y {
                new[x][y] += self;
            }
        }
        new
    }
}

impl<const X: usize, const Y: usize> Sub<Mat<X, Y>> for f32 {
    type Output = Mat<X, Y>;

    fn sub(self, rhs: Mat<X, Y>) -> Self::Output {
        let mut new = rhs.clone();
        for x in 0..X {
            for y in 0..Y {
                new[x][y] -= self;
            }
        }
        new
    }
}

impl<const X: usize, const Y: usize> Sub<f32> for Mat<X, Y> {
    type Output = Self;

    fn sub(self, rhs: f32) -> Self::Output {
        let mut new = self.clone();
        for x in 0..X {
            for y in 0..Y {
                new[x][y] -= rhs;
            }
        }
        new
    }
}

impl<const X: usize, const Y: usize> Mul<Mat<X, Y>> for f32 {
    type Output = Mat<X, Y>;

    fn mul(self, rhs: Mat<X, Y>) -> Self::Output {
        let mut new = rhs.clone();
        for x in 0..X {
            for y in 0..Y {
                new[x][y] *= self;
            }
        }
        new
    }
}

impl<const X: usize, const Y: usize> Mul<f32> for Mat<X, Y> {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        let mut new = self.clone();
        for x in 0..X {
            for y in 0..Y {
                new[x][y] *= rhs;
            }
        }
        new
    }
}

impl<const X: usize, const Y: usize> Div<f32> for Mat<X, Y> {
    type Output = Self;

    fn div(self, rhs: f32) -> Self::Output {
        let mut new = self.clone();
        for x in 0..X {
            for y in 0..Y {
                new[x][y] /= rhs;
            }
        }
        new
    }
}

impl<const X: usize, const Y: usize, const Y2: usize> Mul<Mat<Y, Y2>> for Mat<X, Y> {
    type Output = Mat<X, Y2>;
    // [[f32; Y]; X] * [[f32; Y2]; Y] = [[f32; Y2]; X]
    // self[X][Y] * rhs[Y][Y2] = new[X][Y2]
    fn mul(self, rhs: Mat<Y, Y2>) -> Self::Output {
        let mut new = Mat::default();
        for x in 0..X {
            for y in 0..Y2 {
                for i in 0..Y {
                    new[x][y] += self[x][i] * rhs[i][y];
                }
            }
        }
        new
    }
}
