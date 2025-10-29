// Vect: a vector of f32
// Mat: a matrix of f32
use std::ops::*;

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

            // dot product
            pub fn dot(&self, rhs: $name) -> $type {
                self.0
                    .iter()
                    .zip(rhs.0.iter())
                    .fold(Default::default(), |acc, x| acc + *x.0 * *x.1)
            }

            // iterator conversions
            pub fn iter(&'_ self) -> std::slice::Iter<'_, f32> {
                self.0.iter()
            }

            pub fn iter_mut(&'_ mut self) -> std::slice::IterMut<'_, f32> {
                self.0.iter_mut()
            }

            pub fn into_iter(self) -> std::array::IntoIter<f32, $size> {
                self.0.into_iter()
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
                $name::from(self.0)
            }
        }
    };

    ($name:ident => $trait_name:ident ($trait_fn:ident), $op_by:tt) => {
        impl $trait_name<$name> for $name {
            type Output = $name;

            fn $trait_fn(mut self, rhs: $name) -> $name {
                for x in 0..Self::LEN {
                    self[x] = self[x] $op_by rhs[x];
                }
                $name::from(self.0)
            }
        }
    };
}

vec_gen!(Vec2 => 2, f32);
vec_gen!(Vec3 => 3, f32);
vec_gen!(Vec4 => 4, f32);

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

// Marker trait for "can Index up to four fields" (X, Y, Z, W)
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a indexable struct, or does the group does not match"
)]
pub trait GroupIdx4 {
    type Group;

    fn idx(self) -> usize;
}

// Marker trait for "can Index up to three fields" (X, Y, Z)
#[diagnostic::on_unimplemented(
    message = "`{Self}` is indexing into a \"vec\" whose size is 3 but `{Self}` expects a greater size"
)]
pub trait GroupIdx3: GroupIdx4 {}
// Marker trait for "can Index up to two fields" (X, Y)
#[diagnostic::on_unimplemented(
    message = "`{Self}` is indexing into a \"vec\" whose size is 2 but `{Self}` expects a greater size"
)]
pub trait GroupIdx2: GroupIdx3 {}

// Marker trait for "This type has only indexes that can do the first three elements"
#[diagnostic::on_unimplemented(
    message = "`{Self}` does not have indexing elements less than three",
    note = "in a four component group (ABCD), the first two elements (ABC) are valid for this"
)]
pub trait ContainsOnlyIdx3 {}

// Marker trait for "This type has only indexes that can do the first two elements"
#[diagnostic::on_unimplemented(
    message = "`{Self}` does not have indexing elements less than two",
    note = "in a four component group (ABCD), the first two elements (AB) are valid for this"
)]
pub trait ContainsOnlyIdx2 {}

// There is not a ContainsOnlyIdx4, as all structs that would contain idx elements
// will contain GroupIdx4 only

// Structs to model Indexing
pub struct VecIdx2<One, Two>(One, Two)
where
    One: GroupIdx4,
    // this pattern here is repeated a lot, tl;dr it makes sure that the group
    // is the same for each idx type
    Two: GroupIdx4<Group = One::Group>;

impl<One, Two> VecIdx2<One, Two>
where
    One: GroupIdx4,
    Two: GroupIdx4<Group = One::Group>,
{
    // create a new fn
    pub fn new(lhs: One, rhs: Two) -> Self {
        VecIdx2(lhs, rhs)
    }

    // each of these has an associated vec size that can be filled
    fn assoc_vec(&self) -> Vec2 {
        Vec2::default()
    }

    // deconstruct it to indexes
    pub fn decompose(self) -> [usize; 2] {
        [self.0.idx(), self.1.idx()]
    }
}

// following two checks are for if the digits are of a certain, lower size
impl<One, Two> ContainsOnlyIdx3 for VecIdx2<One, Two>
where
    One: GroupIdx4 + GroupIdx3,
    Two: GroupIdx4<Group = One::Group> + GroupIdx3,
{
}

impl<One, Two> ContainsOnlyIdx2 for VecIdx2<One, Two>
where
    One: GroupIdx4 + GroupIdx3 + GroupIdx2,
    Two: GroupIdx4<Group = One::Group> + GroupIdx3 + GroupIdx2,
{
}

// repeat above for idx3
pub struct VecIdx3<One, Two, Three>(One, Two, Three)
where
    One: GroupIdx4,
    Two: GroupIdx4<Group = One::Group>,
    Three: GroupIdx4<Group = Two::Group>;

impl<One, Two, Three> VecIdx3<One, Two, Three>
where
    One: GroupIdx4,
    Two: GroupIdx4<Group = One::Group>,
    Three: GroupIdx4<Group = Two::Group>,
{
    pub fn new(lhs: VecIdx2<One, Two>, rhs: Three) -> Self {
        VecIdx3(lhs.0, lhs.1, rhs)
    }

    fn assoc_vec(&self) -> Vec3 {
        Vec3::default()
    }

    pub fn decompose(self) -> [usize; 3] {
        [self.0.idx(), self.1.idx(), self.2.idx()]
    }
}

impl<One, Two, Three> ContainsOnlyIdx3 for VecIdx3<One, Two, Three>
where
    One: GroupIdx4 + GroupIdx3,
    Two: GroupIdx4<Group = One::Group> + GroupIdx3,
    Three: GroupIdx4<Group = Two::Group> + GroupIdx3,
{
}

impl<One, Two, Three> ContainsOnlyIdx2 for VecIdx3<One, Two, Three>
where
    One: GroupIdx4 + GroupIdx3 + GroupIdx2,
    Two: GroupIdx4<Group = One::Group> + GroupIdx3 + GroupIdx2,
    Three: GroupIdx4<Group = One::Group> + GroupIdx3 + GroupIdx2,
{
}

// repeat above for idx4
pub struct VecIdx4<One, Two, Three, Four>(One, Two, Three, Four)
where
    One: GroupIdx4,
    Two: GroupIdx4<Group = One::Group>,
    Three: GroupIdx4<Group = Two::Group>,
    Four: GroupIdx4<Group = Three::Group>;

impl<One, Two, Three, Four> VecIdx4<One, Two, Three, Four>
where
    One: GroupIdx4,
    Two: GroupIdx4<Group = One::Group>,
    Three: GroupIdx4<Group = Two::Group>,
    Four: GroupIdx4<Group = Three::Group>,
{
    pub fn new(lhs: VecIdx3<One, Two, Three>, rhs: Four) -> Self {
        VecIdx4(lhs.0, lhs.1, lhs.2, rhs)
    }

    fn assoc_vec(&self) -> Vec4 {
        Vec4::default()
    }

    pub fn decompose(self) -> [usize; 4] {
        [self.0.idx(), self.1.idx(), self.2.idx(), self.3.idx()]
    }
}

impl<One, Two, Three, Four> ContainsOnlyIdx3 for VecIdx4<One, Two, Three, Four>
where
    One: GroupIdx4 + GroupIdx3,
    Two: GroupIdx4<Group = One::Group> + GroupIdx3,
    Three: GroupIdx4<Group = Two::Group> + GroupIdx3,
    Four: GroupIdx4<Group = Three::Group> + GroupIdx3,
{
}

impl<One, Two, Three, Four> ContainsOnlyIdx2 for VecIdx4<One, Two, Three, Four>
where
    One: GroupIdx4 + GroupIdx3 + GroupIdx2,
    Two: GroupIdx4<Group = One::Group> + GroupIdx3 + GroupIdx2,
    Three: GroupIdx4<Group = Two::Group> + GroupIdx3 + GroupIdx2,
    Four: GroupIdx4<Group = Three::Group> + GroupIdx3 + GroupIdx2,
{
}

macro_rules! swiz_panic {
    ($idx:ident (One, $($gen:ident),*): $typ:ident) => {
        impl<One, $($gen),*> Index<$idx<One, $($gen),*>> for $typ
        where
            One: GroupIdx4,
            $($gen: GroupIdx4<Group = One::Group>),*
        {
            type Output = ();

            fn index(&self, _: $idx<One, $($gen),*>) -> &() {
                const {
                    panic!(
                        "you cannot swizzle a Vec directly, use the `swz!` and `swz_assign!` macro instead"
                    );
                }
                // This is intentional, as the previous line will always panic if this function is called.
                // This is just to pass typechecking, because the real use of this function is to smuggle
                // that const panic.
                #[allow(unreachable_code)]
                &()
            }
        }
    };
}

swiz_panic!(VecIdx2 (One, Two): Vec2);
swiz_panic!(VecIdx2 (One, Two): Vec3);
swiz_panic!(VecIdx2 (One, Two): Vec4);
swiz_panic!(VecIdx3 (One, Two, Three): Vec2);
swiz_panic!(VecIdx3 (One, Two, Three): Vec3);
swiz_panic!(VecIdx3 (One, Two, Three): Vec4);
swiz_panic!(VecIdx4 (One, Two, Three, Four): Vec2);
swiz_panic!(VecIdx4 (One, Two, Three, Four): Vec3);
swiz_panic!(VecIdx4 (One, Two, Three, Four): Vec4);

// index struct generation
macro_rules! index_gen {
    (($group:ident): $one:ident, $two:ident, $three:ident, $four:ident) => {
        #[derive(Clone, Copy, Debug)]
        pub struct $one;
        #[derive(Clone, Copy, Debug)]
        pub struct $two;
        #[derive(Clone, Copy, Debug)]
        pub struct $three;
        #[derive(Clone, Copy, Debug)]
        pub struct $four;

        #[derive(Clone, Copy, Debug)]
        pub struct $group;

        impl GroupIdx4 for $one {
            type Group = $group;
            #[inline(always)]
            fn idx(self) -> usize { 0 }
        }

        impl GroupIdx4 for $two {
            type Group = $group;
            #[inline(always)]
            fn idx(self) -> usize { 1 }
        }

        impl GroupIdx4 for $three {
            type Group = $group;
            #[inline(always)]
            fn idx(self) -> usize { 2 }
        }

        impl GroupIdx4 for $four {
            type Group = $group;
            #[inline(always)]
            fn idx(self) -> usize { 3 }
        }

        impl GroupIdx3 for $one {}

        impl GroupIdx3 for $two {}

        impl GroupIdx3 for $three {}

        impl GroupIdx2 for $one {}

        impl GroupIdx2 for $two {}

        index_bitor!($one $two $three $four);
    };
}

// ZST to init macro repeats to bitor all structs together
struct NullZST;

macro_rules! index_bitor {
    ($sing:ident) => {
        // Self struct
        impl BitOr for $sing {
            type Output = VecIdx2<$sing, $sing>;

            fn bitor(self, rhs: $sing) -> VecIdx2<$sing, $sing> {
                VecIdx2::new(self, rhs)
            }
        }

        impl BitOr<$sing> for NullZST {
            type Output = $sing;

            fn bitor(self, rhs: $sing) -> $sing {
                rhs
            }
        }
    };

    ($lhs:ident $($rhs:ident)*) => {
        // bitor self and NullZST
        index_bitor!($lhs);
        // recurse: bitor the next items
        index_bitor!($($rhs)*);

        $(
            // the actual repeat impls
            impl BitOr<$rhs> for $lhs {
                type Output = VecIdx2<$lhs, $rhs>;

                fn bitor(self, rhs: $rhs) -> VecIdx2<$lhs, $rhs> {
                    VecIdx2::new(self, rhs)
                }
            }

            impl BitOr<$lhs> for $rhs {
                type Output = VecIdx2<$rhs, $lhs>;

                fn bitor(self, rhs: $lhs) -> VecIdx2<$rhs, $lhs> {
                    VecIdx2::new(self, rhs)
                }
            }
        )*
    };
}

index_gen!((XYZW): X, Y, Z, W);
index_gen!((RGBA): R, G, B, A);
index_gen!((STPQ): S, T, P, Q);

impl<T> Index<T> for Vec2
where
    T: GroupIdx4 + GroupIdx3 + GroupIdx2,
{
    type Output = f32;

    #[inline(always)]
    fn index(&self, index: T) -> &Self::Output {
        &self[index.idx()]
    }
}

impl<T> IndexMut<T> for Vec2
where
    T: GroupIdx4 + GroupIdx3 + GroupIdx2,
{
    #[inline(always)]
    fn index_mut(&mut self, index: T) -> &mut Self::Output {
        &mut self[index.idx()]
    }
}

impl<T> Index<T> for Vec3
where
    T: GroupIdx4 + GroupIdx3,
{
    type Output = f32;

    #[inline(always)]
    fn index(&self, index: T) -> &Self::Output {
        &self[index.idx()]
    }
}

impl<T> IndexMut<T> for Vec3
where
    T: GroupIdx4 + GroupIdx3,
{
    #[inline(always)]
    fn index_mut(&mut self, index: T) -> &mut Self::Output {
        &mut self[index.idx()]
    }
}

impl<T> Index<T> for Vec4
where
    T: GroupIdx4,
{
    type Output = f32;

    #[inline(always)]
    fn index(&self, index: T) -> &Self::Output {
        &self[index.idx()]
    }
}

impl<T> IndexMut<T> for Vec4
where
    T: GroupIdx4,
{
    #[inline(always)]
    fn index_mut(&mut self, index: T) -> &mut Self::Output {
        &mut self[index.idx()]
    }
}

// some specific vec defs for swizzling checks below
impl Vec2 {
    // This function fails if the params are greater than what's indexable
    fn check_idx<T: ContainsOnlyIdx2>(&self, _: &T) {}

    // This does the previous function in addition to checking if the two parameters
    // are in the same group.
    fn check_group_and_idx<One, Two>(&self, i: &VecIdx2<One, Two>)
    where
        One: GroupIdx4 + GroupIdx3 + GroupIdx2,
        Two: GroupIdx4<Group = One::Group> + GroupIdx3 + GroupIdx2,
    {
        self.check_idx(i)
    }
}

impl Vec3 {
    fn check_idx<T: ContainsOnlyIdx3>(&self, _: &T) {}

    fn check_group_and_idx<One, Two, Three>(&self, i: &VecIdx3<One, Two, Three>)
    where
        One: GroupIdx4 + GroupIdx3 + GroupIdx2,
        Two: GroupIdx4<Group = One::Group> + GroupIdx3 + GroupIdx2,
        Three: GroupIdx4<Group = Two::Group> + GroupIdx3 + GroupIdx2,
    {
        self.check_idx(i)
    }
}

impl Vec4 {
    fn check_idx<T>(&self, _: &T) {}

    fn check_group_and_idx<One, Two, Three, Four>(&self, i: &VecIdx4<One, Two, Three, Four>)
    where
        One: GroupIdx4 + GroupIdx3 + GroupIdx2,
        Two: GroupIdx4<Group = One::Group> + GroupIdx3 + GroupIdx2,
        Three: GroupIdx4<Group = Two::Group> + GroupIdx3 + GroupIdx2,
        Four: GroupIdx4<Group = Three::Group> + GroupIdx3 + GroupIdx2,
    {
        self.check_idx(i)
    }
}

// Why swizzling alone needs a macro
// TL;DR: Index returns a ref, and I don't know of a way
// to sucessfully return that ref. So, some private functions are
// created and then typechecked together.

#[macro_export]
macro_rules! swz {
    ($lhs:ident[$($flag:ident)|*]) => {
        // construct idx type normally
        let decompose_group = NullZST $(| $flag)*;
        // make associated vec size
        let mut newvec = decompose_group.assoc_vec();
        // then do check if vec cannot be indexed higher (eg: only XY on a Vec2)
        $lhs.check_idx(&decompose_group);
        // finally, construct newvec
        for (e, i) in decompose_group.decompose().into_iter().enumerate() {
            newvec[e] = $lhs[i];
        }
        newvec
    };
}

// Why does assign need a macro?
// Because Rust doesn't allow assign to be overridden.
#[macro_export]
macro_rules! swz_assign {
    (*$lhs:ident[$($flag:ident)|*] = $rhs:expr) => {
        // construct idx type normally
        let decompose_group = NullZST $(| $flag)*;
        // make associated vec size
        let mut newvec = decompose_group.assoc_vec();
        // then do typecheck if vec cannot be indexed higher (eg: only XY on a Vec2)
        // and that it also is of a specific type
        newvec.check_group_and_idx(&decompose_group);
        // finally, assign each item
        for (e, x) in decompose_group.decompose().into_iter().zip($rhs) {
            $lhs[e] = $rhs;
        }
    };
}

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

        impl From<[[$type; $vec_row::LEN]; $col_len]> for $name {
            fn from(i: [[$type; $vec_row::LEN]; $col_len]) -> Self {
                let mut new: [$vec_row; $col_len] = Default::default();
                for x in 0..i.len() {
                    new[x] = $vec_row::from(i[x]);
                }
                Self::from(new)
            }
        }

        // indexing, OF NOTE: indexing is COLS first
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
                        new[row] += self[row][col] * rhs[col];
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
                for i in 0..Self::COLS {
                    ret[i][i] = 1.0;
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
                // invert I over the selection and you have the scalar.
                self[Self::I - x][Self::I - y] * (if (x + y) % 2 == 1 { -1. } else { 1. })
            }
        }
    };
}

cofactor_mat!(Mat2(f32));
cofactor_mat!(Mat3(f32) > Mat2);
cofactor_mat!(Mat4(f32) > Mat3);
