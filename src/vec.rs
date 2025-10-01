// Vect: a vector of f32
// Mat: a matrix of f32
use generic_array::*;
use std::ops::*;
use typenum::consts::*;
use typenum::*;

#[repr(transparent)]
#[derive(Debug, Clone)]
struct Vect<I: ArrayLength + Unsigned> {
    vec: GenericArray<f32, I>,
}

impl<I: ArrayLength + Unsigned> Vect<I> {
    pub fn new() -> Self {
        const {
            assert!(<I as Unsigned>::USIZE < 5 && <I as Unsigned>::USIZE > 0);
        }
        Self {
            vec: GenericArray::default(),
        }
    }

    pub fn dot(&self, rhs: &Vect<I>) -> f32 {
        self.vec
            .iter()
            .zip(rhs.vec.iter())
            .fold(0.0f32, |acc, x| acc + x.0 * x.1)
    }
}

impl<I: ArrayLength + Unsigned> Default for Vect<I> {
    fn default() -> Self {
        Self::new()
    }
}

// indexing
impl<I: ArrayLength + Unsigned> Index<usize> for Vect<I> {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        &self.vec[index]
    }
}

impl<I: ArrayLength + Unsigned> IndexMut<usize> for Vect<I> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.vec[index]
    }
}

// well defined operations
impl<I: ArrayLength + Unsigned> Add<Vect<I>> for Vect<I> {
    type Output = Vect<I>;

    fn add(mut self, rhs: Vect<I>) -> Self::Output {
        for x in 0..I::USIZE {
            self[x] += rhs[x];
        }
        self
    }
}

impl<I: ArrayLength + Unsigned> Sub<Vect<I>> for Vect<I> {
    type Output = Vect<I>;

    fn sub(mut self, rhs: Vect<I>) -> Self::Output {
        for x in 0..I::USIZE {
            self[x] -= rhs[x];
        }
        self
    }
}

// broadcast ops
impl<I: ArrayLength + Unsigned> Add<f32> for Vect<I> {
    type Output = Vect<I>;

    fn add(mut self, rhs: f32) -> Self::Output {
        for x in 0..I::USIZE {
            self[x] += rhs;
        }
        self
    }
}

impl<I: ArrayLength + Unsigned> Sub<f32> for Vect<I> {
    type Output = Vect<I>;

    fn sub(mut self, rhs: f32) -> Self::Output {
        for x in 0..I::USIZE {
            self[x] -= rhs;
        }
        self
    }
}

impl<I: ArrayLength + Unsigned> Mul<f32> for Vect<I> {
    type Output = Vect<I>;

    fn mul(mut self, rhs: f32) -> Self::Output {
        for x in 0..I::USIZE {
            self[x] *= rhs;
        }
        self
    }
}

impl<I: ArrayLength + Unsigned> Div<f32> for Vect<I> {
    type Output = Vect<I>;

    fn div(mut self, rhs: f32) -> Self::Output {
        for x in 0..I::USIZE {
            self[x] /= rhs;
        }
        self
    }
}

#[repr(transparent)]
#[derive(Debug, Clone)]
struct Mat<X: ArrayLength + Unsigned, Y: ArrayLength + Unsigned> {
    mat: GenericArray<Vect<Y>, X>,
}

impl<X: ArrayLength + Unsigned, Y: ArrayLength + Unsigned> Mat<X, Y> {
    pub fn new() -> Self {
        const {
            assert!(<X as Unsigned>::USIZE < 5 && <X as Unsigned>::USIZE > 0);
            assert!(<Y as Unsigned>::USIZE < 5 && <Y as Unsigned>::USIZE > 0);
        }
        Self {
            mat: GenericArray::default(),
        }
    }

    pub fn t(self) -> Mat<Y, X> {
        let mut ret = Mat::new();
        for x in 0..X::USIZE {
            for y in 0..Y::USIZE {
                ret[y][x] = self[x][y];
            }
        }
        ret
    }
}

impl<X: ArrayLength + Unsigned, Y: ArrayLength + Unsigned> Index<usize> for Mat<X, Y> {
    type Output = Vect<Y>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.mat[index]
    }
}

impl<X: ArrayLength + Unsigned, Y: ArrayLength + Unsigned> IndexMut<usize> for Mat<X, Y> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.mat[index]
    }
}

impl<X: ArrayLength + Unsigned, Y: ArrayLength + Unsigned> Default for Mat<X, Y> {
    fn default() -> Self {
        Self::new()
    }
}

// standard ops
impl<X: ArrayLength + Unsigned, Y: ArrayLength + Unsigned> Add<Mat<X, Y>> for Mat<X,Y> {
    type Output = Mat<X,Y>;

    fn add(mut self, rhs: Mat<X, Y>) -> Self::Output {
        for x in 0..X::USIZE {
            for y in 0..Y::USIZE {
                self[x][y] += rhs[x][y];
            }
        }
        self
    }
}

// matmul
impl<X: ArrayLength + Unsigned, M: ArrayLength + Unsigned, Y: ArrayLength + Unsigned> Mul<Mat<M, Y>>
    for Mat<X, M>
{
    type Output = Mat<X, Y>;

    fn mul(self, rhs: Mat<M, Y>) -> Self::Output {
        let mut ret = Mat::new();
        // [[f32; Y]; X] * [[f32; Y2]; Y] = [[f32; Y2]; X]
        // self[X][Y] * rhs[Y][Y2] = new[X][Y2]
        for x in 0..X::USIZE {
            for y in 0..Y::USIZE {
                for i in 0..M::USIZE {
                    ret[x][y] += self[x][i] * rhs[i][y];
                } 
            }
        }
        ret 
    }
}
