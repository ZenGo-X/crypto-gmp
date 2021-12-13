use std::borrow::Cow;
use std::cmp::Ordering;
use std::ops::{Add, Div, Mul, Rem};

use gmp_mpfr_sys::gmp::size_t;
use subtle::{Choice, ConditionallySelectable};
use zeroize::Zeroize;

use crate::digits::{Digit, U64};

pub mod digits;
mod ops;

pub struct BigInt<D = Box<[Digit]>> {
    digits: D,
}

impl<D> BigInt<D> {
    #[inline(always)]
    pub fn from_digits(digits: D) -> Self {
        Self { digits }
    }
}

impl<D> From<D> for BigInt<D> {
    #[inline(always)]
    fn from(digits: D) -> Self {
        Self { digits }
    }
}

impl<'d> BigInt<Cow<'d, [Digit]>> {
    /// Constructs an integer from u32 digits
    #[inline(always)]
    pub fn from_digits_u32(digits: &'d [u32]) -> Self {
        Self::from_digits(digits::digits_from_limbs_u32(digits))
    }
}

impl<D: AsRef<[Digit]>> BigInt<D> {
    #[inline(always)]
    pub fn to_digits_u32(&self) -> Cow<[u32]> {
        digits::digits_to_limbs_u32(self.as_ref())
    }
}

impl<D: AsRef<[Digit]>> BigInt<D> {
    /// Returns number of digits that represent this integer
    ///
    /// Note that integer can be padded with zeroes, so `a.size() > b.size()` doesn't mean that
    /// `a > b`
    #[inline(always)]
    pub fn size(&self) -> usize {
        self.digits.as_ref().len()
    }

    /// Clones BigInt
    ///
    /// Allocates a new integer with the same size on the heap and copies its value to there
    pub fn reallocate(&self) -> BigInt {
        let mut new_place = vec![Digit::zero(); self.size()];
        new_place.copy_from_slice(self.as_ref());
        BigInt::from_digits(new_place.into_boxed_slice())
    }
}

impl BigInt<[Digit; 1]> {
    #[inline(always)]
    pub fn zero() -> Self {
        BigInt::from_digits([Digit::zero()])
    }
}

macro_rules! impl_digit_operation {
    ($name:ident,$op:ident,$tp:ty) => {
        impl<D: AsRef<[Digit]>> $name<$tp> for &BigInt<D> {
            type Output = BigInt;

            #[inline(always)]
            fn $op(self, rhs: $tp) -> Self::Output {
                self.$op(Digit::from(rhs))
            }
        }
    };
}

macro_rules! impl_digit_operation_reverse {
    ($name:ident,$op:ident,$tp:ty) => {
        impl<'n, D: AsRef<[Digit]>> $name<&'n BigInt<D>> for $tp {
            type Output = BigInt;

            #[inline(always)]
            fn $op(self, rhs: &BigInt<D>) -> Self::Output {
                rhs.$op(self)
            }
        }
    };
}

macro_rules! impl_u64_operation {
    ($name:ident,$op:ident) => {
        impl<D: AsRef<[Digit]>> $name<u64> for &BigInt<D> {
            type Output = BigInt;

            #[inline(always)]
            fn $op(self, rhs: u64) -> Self::Output {
                self.$op(&BigInt::from_digits(&U64::from(rhs)))
            }
        }
    };
}

impl<D1, D2> Add<&BigInt<D2>> for &BigInt<D1>
where
    D1: AsRef<[Digit]>,
    D2: AsRef<[Digit]>,
{
    type Output = BigInt;

    /// Computes `self + rhs`
    ///
    /// ## Constant time
    /// Addition is constant time in size of its arguments. It means that number of instructions
    /// and memory access patterns remain the same for the same `self.size()` and `rhs.size()`
    ///
    /// ## Zeroizing
    /// This function might allocate some auxiliary buffer on heap. Before deallocating, it gets
    /// zeroed out.
    ///
    /// ## Complexity
    /// Addition complexity is `O(max(self.size(), rhs.size()))`
    fn add(self, rhs: &BigInt<D2>) -> Self::Output {
        // We have to maintain a decreasing order of lengths of the input arguments:
        // `a.len() >= b.len()`
        let (a, b) = ops::reorder(self.as_ref(), rhs.as_ref());

        let n = a.len();
        let m = b.len();

        // Get rid of empty slices: gmp code doesn't work with them
        match (n, m) {
            (0, 0) => return BigInt::zero().reallocate(),
            (0, _) => return BigInt::from_digits(b).reallocate(),
            (_, 0) => return BigInt::from_digits(a).reallocate(),
            _ => {}
        }

        // Estimate the space we need to allocate on the heap
        let output_size = n + 1;
        let scratch_space_size = if n == m {
            0
        } else {
            // Safety: `reorder` guarantees that n >= m, being at this branch guarantees
            // that n != m
            unsafe { ops::add_n_m_itch(n, m) }
        };
        debug_assert_eq!(scratch_space_size, ScratchSpace::addition(n, m));

        // Allocate space on heap, split it into `output` that stands for addition result and
        // `scratch_space` that is auxiliary buffer for gmp functions
        let mut buffer = vec![Digit::zero(); output_size + scratch_space_size];
        let (output, scratch_space) = buffer.split_at_mut(output_size);
        let mut output = BigInt::from_digits(output);
        let scratch_space = ScratchSpace::new(scratch_space);

        unsafe {
            // Safety:
            // * `reorder` call above guarantees that `n >= m`
            // * We checked that n and m are not zero (see match statement above)
            // * We allocated sufficient memory for both output and scratch space
            output.add_unchecked(
                &BigInt::from_digits(a),
                &BigInt::from_digits(b),
                scratch_space,
            );
        }

        // Now we need to resize `buffer` to `output_size`, ie. we want to strip the allocated scratch
        // space. But we also wish to check whether the last digit of `output` is zero or not. If it
        // is, we strip it too

        // Note: we are comparing two size_t integers here, we assume it's constant-time
        let last_digit_is_zero = Choice::from(u8::from(output.as_ref()[n] == Digit::zero()));
        let output_size =
            size_t::conditional_select(&((n + 1) as size_t), &(n as size_t), last_digit_is_zero)
                as usize;

        buffer[output_size..].zeroize();
        buffer.resize(output_size, Digit::zero());

        BigInt::from_digits(buffer.into_boxed_slice())
    }
}

impl<D: AsMut<[Digit]>> BigInt<D> {
    /// Computes `a + b`, writes result to `self`
    ///
    /// It is a low-level function allowing you to manage allocations. Its safe analogue is
    /// [`&a + &b`](#impl-Add%3C%26%27_%20BigInt%3CD2%3E%3E)
    ///
    /// ## Safety
    /// Assuming that `a` and `b` have [sizes][size] `n` and `m` respectively:
    /// * `n >= m > 0`
    /// * [Size][size] of `self` must be exactly `n+1` limbs
    /// * Size of `scratch_space` must be exactly [`ScratchSpace::addition(n, m)`](ScratchSpace::addition) limbs
    ///
    /// Note: we check that these requirements are met using debug assertions.
    ///
    /// [size]: BigInt::size
    ///
    /// ## Constant time
    /// This function is constant time in size of its arguments, ie. number of instructions and
    /// memory access patterns remain the same for the same `n` and `m`.
    pub unsafe fn add_unchecked(
        &mut self,
        a: &BigInt<impl AsRef<[Digit]>>,
        b: &BigInt<impl AsRef<[Digit]>>,
        mut scratch_space: ScratchSpace,
    ) {
        let (n, m) = (a.size(), b.size());
        if n == m {
            let carry = ops::add_n(a.as_ref(), b.as_ref(), &mut self.as_mut()[..n]);
            self.as_mut()[n] = carry;
        } else {
            let carry = ops::add_n_m(
                a.as_ref(),
                b.as_ref(),
                scratch_space.as_mut(),
                &mut self.as_mut()[..n],
            );
            self.as_mut()[n] = carry;
        }
    }
}

impl<D: AsRef<[Digit]>> Add<Digit> for &BigInt<D> {
    type Output = BigInt;

    /// Computes `self + rhs` where `rhs` is a `Digit`.
    ///
    /// ## Constant time
    /// Even though adding a single digit might be done in `O(1)` on average,
    /// the number of instructions and memory access patterns of this method
    /// deliberately remains the same for the same `self.size()`.
    ///
    /// ## Zeroizing
    /// This function might allocate some auxiliary buffer on the heap.
    /// Before deallocating, it gets zeroed out.
    ///
    /// ## Complexity
    /// Complexity is always `O(self.size())`.
    fn add(self, rhs: Digit) -> Self::Output {
        // We don't need to reorder operands as `rhs` is a single `Digit`
        let a = self.as_ref();
        let n = a.len();

        // Get rid of empty slices: gmp code doesn't work with them
        if n == 0 {
            return BigInt::from_digits(&[rhs]).reallocate();
        }

        // Estimate the space we need to allocate on the heap
        let output_size = n + 1;
        let scratch_space_size = ScratchSpace::digit_addition(n);

        // Allocate space on the heap, split it into `output` that stands for addition result and
        // `scratch_space` that is auxiliary buffer for gmp functions
        let mut buffer = vec![Digit::zero(); output_size + scratch_space_size];
        let (output, scratch_space) = buffer.split_at_mut(output_size);
        let mut output = BigInt::from_digits(output);
        let scratch_space = ScratchSpace::new(scratch_space);

        unsafe {
            // Safety:
            // * We checked that n is not zero (see match statement above)
            // * We allocated sufficient memory for both output and scratch space
            output.add_digit_unchecked(&BigInt::from_digits(a), rhs, scratch_space);
        };

        // Note: we are comparing two size_t integers here, we assume it's constant-time
        let last_digit_is_zero = Choice::from(u8::from(output.as_ref()[n] == Digit::zero()));
        let output_size =
            size_t::conditional_select(&((n + 1) as size_t), &(n as size_t), last_digit_is_zero)
                as usize;

        // Now we need to resize `buffer` to `output_size`,
        // ie. we want to strip the allocated scratch space.
        buffer[output_size..].zeroize();
        buffer.resize(output_size, Digit::zero());

        BigInt::from_digits(buffer.into_boxed_slice())
    }
}

impl<D: AsMut<[Digit]>> BigInt<D> {
    /// Computes `a + b`, writes result to `self`
    ///
    /// It is a low-level function allowing you to manage allocations. Its safe analogue is
    /// [`&a + b`](#impl-Add%3CDigit%3E).
    ///
    /// ## Safety
    /// Assuming that `a` has [size] `n`:
    /// * `n > 0`
    /// * [Size][size] of `self` must be exactly `n+1` limbs
    /// * Size of `scratch_space` must be exactly
    /// [`ScratchSpace::digit_addition(n)`](ScratchSpace::digit_addition) limbs
    ///
    /// [size]: BigInt::size
    ///
    /// ## Constant time
    /// This function is constant time in size of its arguments, ie. number of instructions and
    /// memory access patterns remain the same for the same `n`.
    pub unsafe fn add_digit_unchecked(
        &mut self,
        a: &BigInt<impl AsRef<[Digit]>>,
        b: Digit,
        mut scratch_space: ScratchSpace,
    ) {
        let n = a.size();
        let carry = ops::add_1(
            a.as_ref(),
            b,
            scratch_space.as_mut(),
            &mut self.as_mut()[..n],
        );
        self.as_mut()[n] = carry;
    }
}

impl_digit_operation!(Add, add, u8);
impl_digit_operation_reverse!(Add, add, u8);
impl_digit_operation!(Add, add, u16);
impl_digit_operation_reverse!(Add, add, u16);
impl_digit_operation!(Add, add, u32);
impl_digit_operation_reverse!(Add, add, u32);
impl_u64_operation!(Add, add);
impl_digit_operation_reverse!(Add, add, u64);
impl_digit_operation_reverse!(Add, add, Digit);

impl<D1, D2> Mul<&BigInt<D2>> for &BigInt<D1>
where
    D1: AsRef<[Digit]>,
    D2: AsRef<[Digit]>,
{
    type Output = BigInt;

    /// Computes `self * rhs`
    ///
    /// ## Constant time
    /// Multiplication is constant time in size of its arguments. It means that the number of
    /// instructions and memory access patterns remain the same for the same `self.size()` and `rhs.size()`
    ///
    /// ## Zeroizing
    /// This function might allocate some auxiliary buffer on heap. Before deallocating, it gets
    /// zeroed out.
    ///
    /// ## Complexity
    /// Fast multiplication is not used, so the complexity is always `O(self.size() * rhs.size())`.
    fn mul(self, rhs: &BigInt<D2>) -> Self::Output {
        // We have to maintain a decreasing order of lengths of the input arguments:
        // `a.len() >= b.len()`
        let (a, b) = ops::reorder(self.as_ref(), rhs.as_ref());

        let n = a.len();
        let m = b.len();

        // Get rid of empty slices: gmp code doesn't work with them
        match (n, m) {
            (0, _) | (_, 0) => return BigInt::zero().reallocate(),
            _ => {}
        }

        // Estimate the space we need to allocate on the heap
        let output_size = n + m;
        let scratch_space_size = unsafe { ops::mul_n_m_itch(n, m) };
        debug_assert_eq!(scratch_space_size, ScratchSpace::multiplication(n, m));

        // Allocate space on the heap, split it into `output` that stands for addition result and
        // `scratch_space` that is auxiliary buffer for gmp functions
        let mut buffer = vec![Digit::zero(); output_size + scratch_space_size];
        let (output, scratch_space) = buffer.split_at_mut(output_size);
        let mut output = BigInt::from_digits(output);
        let scratch_space = ScratchSpace::new(scratch_space);

        unsafe {
            // Safety:
            // * We checked that n>=m and that n and m are non-zero (see match statement above)
            // * We allocated sufficient memory for both output and scratch space
            output.mul_unchecked(
                &BigInt::from_digits(a),
                &BigInt::from_digits(b),
                scratch_space,
            );
        };

        // Note: we are comparing two size_t integers here, we assume it's constant-time
        let last_digit_is_zero =
            Choice::from(u8::from(output.as_ref()[n + m - 1] == Digit::zero()));
        let output_size = size_t::conditional_select(
            &((n + m) as size_t),
            &((n + m - 1) as size_t),
            last_digit_is_zero,
        ) as usize;

        // Now we need to resize `buffer` to `output_size`,
        // ie. we want to strip the allocated scratch space.
        buffer[output_size..].zeroize();
        buffer.resize(output_size, Digit::zero());

        BigInt::from_digits(buffer.into_boxed_slice())
    }
}

impl<D: AsMut<[Digit]>> BigInt<D> {
    /// Computes `a * b`, writes result to `self`
    ///
    /// It is a low-level function allowing you to manage allocations. Its safe analogue is
    /// [`&a * &b`](#impl-Mul%3C%26%27_%20BigInt%3CD2%3E%3E).
    ///
    /// ## Safety
    /// Assuming that `a` and `b` have [sizes][size] `n` and `m` respectively:
    /// * `n >= m > 0`
    /// * [Size][size] of `self` must be exactly `n+m` limbs
    /// * Size of `scratch_space` must be exactly
    /// [`ScratchSpace::multiplication(n, m)`](ScratchSpace::multiplication) limbs
    ///
    /// Note: we check that these requirements are met using debug assertions.
    ///
    /// [size]: BigInt::size
    ///
    /// ## Constant time
    /// This function is constant time in size of its arguments, ie. number of instructions and
    /// memory access patterns remain the same for the same `n` and `m`.
    pub unsafe fn mul_unchecked(
        &mut self,
        a: &BigInt<impl AsRef<[Digit]>>,
        b: &BigInt<impl AsRef<[Digit]>>,
        mut scratch_space: ScratchSpace,
    ) {
        ops::mul_n_m(
            a.as_ref(),
            b.as_ref(),
            scratch_space.as_mut(),
            self.as_mut(),
        );
    }
}

impl<D: AsRef<[Digit]>> Mul<Digit> for &BigInt<D> {
    type Output = BigInt;

    /// Computes `self * rhs` where `rhs` is a `Digit`.
    ///
    /// ## Constant time
    /// Irrespective of it's inputs, this method takes `O(self.size())` steps.
    ///
    /// ## Zeroizing
    /// This function might allocate some auxiliary buffer on the heap.
    /// Before deallocating, it gets zeroed out.
    ///
    /// ## Complexity
    /// The complexity is `O(self.size())`, which is asymptotically optimal.
    fn mul(self, rhs: Digit) -> Self::Output {
        self * &BigInt::from_digits(&[rhs])
    }
}

impl<D: AsMut<[Digit]>> BigInt<D> {
    /// Computes `a * b` where `b` is a digit, writes the result to `self`.
    ///
    /// It is a low-level function allowing you to manage allocations. Its safe analogue is
    /// [`&a * b`](#impl-Mul%3CDigit%3E)
    ///
    /// ## Safety
    /// Assuming that `a` has [size] `n`:
    /// * `n > 0`
    /// * [Size][size] of `self` must be exactly `n+1` limbs
    /// * Size of `scratch_space` must be exactly
    /// [`ScratchSpace::multiplication(n, 1)`](ScratchSpace::multiplication) limbs
    ///
    /// [size]: BigInt::size
    ///
    /// ## Constant time
    /// This function is constant time in size of its arguments, ie. number of instructions and
    /// memory access patterns remain the same for the same `n`.
    pub unsafe fn mul_digit_unchecked(
        &mut self,
        a: &BigInt<impl AsRef<[Digit]>>,
        b: Digit,
        mut scratch_space: ScratchSpace,
    ) {
        self.mul_unchecked(a, &BigInt::from_digits(&[b]), scratch_space);
    }
}

impl_digit_operation!(Mul, mul, u8);
impl_digit_operation_reverse!(Mul, mul, u8);
impl_digit_operation!(Mul, mul, u16);
impl_digit_operation_reverse!(Mul, mul, u16);
impl_digit_operation!(Mul, mul, u32);
impl_digit_operation_reverse!(Mul, mul, u32);
impl_u64_operation!(Mul, mul);
impl_digit_operation_reverse!(Mul, mul, u64);
impl_digit_operation_reverse!(Mul, mul, Digit);

impl<D1, D2> Div<&BigInt<D2>> for &BigInt<D1>
where
    D1: AsRef<[Digit]>,
    D2: AsRef<[Digit]>,
{
    type Output = BigInt;

    /// Computes `floor(self / rhs)`
    ///
    /// ## Constant time
    /// Division is constant time in size of its arguments. It means that the number of
    /// instructions and memory access patterns remain the same for the same
    /// `self.size()` and `rhs.size()` (provided that `rhs` does not start from 0).
    ///
    /// ## Zeroizing
    /// This function might allocate some auxiliary buffer on the heap.
    /// Before deallocating, it gets zeroed out.
    ///
    /// ## Complexity
    /// The complexity is always `O(self.size() * rhs.size())`.
    ///
    /// ## Panics
    /// The method panics if `rhs` equals zero.
    fn div(self, rhs: &BigInt<D2>) -> Self::Output {
        let n = self.size();
        // we have to clone `self` in order to be able to use unchecked division,
        // as it requires its first argument to be mutable
        let mut a = self.reallocate();
        let (a, b) = (a.as_mut(), rhs.as_ref());

        // m is the length of b without leading zeroes
        let m = ops::len_without_leading_zeroes(b);

        if m == 0 {
            panic!("Division by zero");
        }

        // if m > n, then definitely b > a and so floor(a/b) = 0.
        if m > n {
            a.zeroize();
            return BigInt::zero().reallocate();
        }

        // Estimate the space we need to allocate on the heap
        let output_size = n - m + 1;
        let scratch_space_size = unsafe { ops::div_n_m_itch(n, m) };
        debug_assert_eq!(scratch_space_size, ScratchSpace::division(n, m));

        // Allocate space on the heap, split it into `output` that stands for addition result and
        // `scratch_space` that is auxiliary buffer for gmp functions
        let mut buffer = vec![Digit::zero(); output_size + scratch_space_size];
        let (output, scratch_space) = buffer.split_at_mut(output_size);
        let mut output = BigInt::from_digits(output);
        let scratch_space = ScratchSpace::new(scratch_space);

        let mut a_bigint = BigInt::from_digits(a);
        unsafe {
            // Safety:
            // * We checked that n>=m and that n and m are non-zero (see match statement above)
            // * We allocated sufficient memory for both output and scratch space
            output.div_unchecked(&mut a_bigint, &BigInt::from_digits(&b[..m]), scratch_space);
        };

        // Note: we are comparing two size_t integers here, we assume it's constant-time
        let last_digit_is_zero = Choice::from(u8::from(output.as_ref()[n - m] == Digit::zero()));
        let output_size = size_t::conditional_select(
            &((n - m + 1) as size_t),
            &((n - m) as size_t),
            last_digit_is_zero,
        ) as usize;

        // Now we need to resize `buffer` to `output_size`,
        // ie. we want to strip the allocated scratch space.
        buffer[output_size..].zeroize();
        buffer.resize(output_size, Digit::zero());
        // Also zeroize `a` that has been allocated earlier
        a_bigint.as_mut().zeroize();

        BigInt::from_digits(buffer.into_boxed_slice())
    }
}

impl<D: AsMut<[Digit]>> BigInt<D> {
    /// Computes `floor(a / b)`, writes the result to `self` and remainder `a mod b` - to
    /// the `len(b)` most significant limbs of `a`. Accordingly, `a` must be mutable.
    ///
    /// It is a low-level function allowing you to manage allocations. Its safe analogue is
    /// [`&a / &b`](#impl-Div%3C%26%27_%20BigInt%3CD2%3E%3E).
    ///
    /// ## Safety
    /// Assuming that `a` and `b` have [sizes][size] `n` and `m` respectively:
    /// * `n >= m > 0`
    /// * [Size][size] of `self` must be exactly `n-m` limbs
    /// * Size of `scratch_space` must be exactly
    /// [`ScratchSpace::division(n, m)`](ScratchSpace::division) limbs
    ///
    /// Note: we check that these requirements are met using debug assertions.
    ///
    /// [size]: BigInt::size
    ///
    /// ## Constant time
    /// This function is constant time in size of its arguments, ie. number of instructions and
    /// memory access patterns remain the same for the same `n` and `m`.
    pub unsafe fn div_unchecked(
        &mut self,
        a: &mut BigInt<impl AsMut<[Digit]>>,
        b: &BigInt<impl AsRef<[Digit]>>,
        mut scratch_space: ScratchSpace,
    ) {
        let n = a.as_mut().len() - b.size();
        let carry = ops::div_n_m(
            a.as_mut(),
            b.as_ref(),
            scratch_space.as_mut(),
            &mut self.as_mut()[..n],
        );
        self.as_mut()[n] = carry;
    }
}

impl<D: AsRef<[Digit]>> Div<Digit> for &BigInt<D> {
    type Output = BigInt;

    /// Computes `self / rhs` where `rhs` is a `Digit`.
    ///
    /// ## Constant time
    /// Irrespective of it's inputs, this method takes `O(self.size())` steps.
    ///
    /// ## Zeroizing
    /// This function might allocate some auxiliary buffer on the heap.
    /// Before deallocating, it gets zeroed out.
    ///
    /// ## Complexity
    /// The complexity is `O(self.size())`, which is asymptotically optimal.
    ///
    /// ## Panics
    /// The method panics if `rhs` equals zero.
    fn div(self, rhs: Digit) -> Self::Output {
        self / &BigInt::from_digits(&[rhs])
    }
}

impl<D: AsMut<[Digit]>> BigInt<D> {
    /// Computes `floor(a / b)` where `b` is a digit, writes the result to `self`.
    ///
    /// It is a low-level function allowing you to manage allocations. Its safe analogue is
    /// [`&a / b`](#impl-Div%3CDigit%3E).
    ///
    /// ## Safety
    /// Assuming that `a` has [size] `n`:
    /// * `n > 0`
    /// * [Size][size] of `self` must be exactly `n` limbs
    /// * Size of `scratch_space` must be exactly
    /// [`ScratchSpace::division(n, 1)`](ScratchSpace::division) limbs
    ///
    /// [size]: BigInt::size
    ///
    /// ## Constant time
    /// This function is constant time in size of its arguments, ie. number of instructions and
    /// memory access patterns remain the same for the same `n`.
    pub unsafe fn div_digit_unchecked(
        &mut self,
        a: &mut BigInt<impl AsMut<[Digit]>>,
        b: Digit,
        mut scratch_space: ScratchSpace,
    ) {
        self.div_unchecked(a, &BigInt::from_digits(&[b]), scratch_space);
    }
}

impl_digit_operation!(Div, div, u8);
impl_digit_operation_reverse!(Div, div, u8);
impl_digit_operation!(Div, div, u16);
impl_digit_operation_reverse!(Div, div, u16);
impl_digit_operation!(Div, div, u32);
impl_digit_operation_reverse!(Div, div, u32);
impl_u64_operation!(Div, div);
impl_digit_operation_reverse!(Div, div, u64);
impl_digit_operation_reverse!(Div, div, Digit);

impl<D1, D2> Rem<&BigInt<D2>> for &BigInt<D1>
where
    D1: AsRef<[Digit]>,
    D2: AsRef<[Digit]>,
{
    type Output = BigInt;

    /// Computes `self mod rhs`. In other words, the remainder of dividing `self` by `rhs`.
    ///
    /// ## Constant time
    /// Division with remainder is constant time in size of its arguments.
    /// It means that the number of instructions and memory access patterns remain the same for the same
    /// `self.size()` and `rhs.size()` (provided that `rhs` does not start from 0).
    ///
    /// ## Zeroizing
    /// This function might allocate some auxiliary buffer on the heap.
    /// Before deallocating, it gets zeroed out.
    ///
    /// ## Complexity
    /// The complexity is always `O(self.size() * rhs.size())`.
    ///
    /// ## Panics
    /// The method panics if `rhs` equals zero.
    fn rem(self, rhs: &BigInt<D2>) -> Self::Output {
        let n = self.size();

        let b = rhs.as_ref();
        // m is the length of b without leading zeroes
        let m = ops::len_without_leading_zeroes(b);

        if m == 0 {
            panic!("Division by zero");
        }

        // if m > n, then definitely b > a and so a mod b = a.
        if m > n {
            return BigInt::from_digits(self.as_ref()).reallocate();
        }

        // we have to clone `self` in order to be able to use unchecked division with remainder,
        // as it requires its first argument to be mutable
        let mut a_copy = vec![Digit::zero(); n];
        a_copy.copy_from_slice(self.as_ref());

        // Estimate the space we need to allocate on the heap
        let scratch_space_size = unsafe { ops::mod_n_m_itch(n, m) };
        debug_assert_eq!(scratch_space_size, ScratchSpace::division_remainder(n, m));

        // Allocate space on the heap, split it into `output` that stands for addition result and
        // `scratch_space` that is auxiliary buffer for gmp functions
        let mut buffer = vec![Digit::zero(); scratch_space_size];
        let scratch_space = ScratchSpace::new(&mut buffer);

        let mut a_bigint = BigInt::from_digits(&mut a_copy[..]);
        unsafe {
            // Safety:
            // * We checked that n>=m and that n and m are non-zero (see match statement above)
            // * We allocated sufficient memory for the scratch space
            a_bigint.rem_unchecked(&BigInt::from_digits(&b[..m]), scratch_space);
        };

        let output_size = m;
        // we want to zeroize the space allocated for `a_bigint` after the m limbs of the result.
        a_copy[output_size..].zeroize();
        a_copy.resize(output_size, Digit::zero());
        // also zeroizing scratch space
        buffer.zeroize();

        BigInt::from_digits(a_copy.into_boxed_slice())
    }
}

impl<D: AsMut<[Digit]>> BigInt<D> {
    /// Computes `self mod a`, writes the result to the `len(b)` most significant limbs of `self`.
    /// Accordingly, `a` must be mutable.
    ///
    /// It is a low-level function allowing you to manage allocations. Its safe analogue is
    /// [`&self % &a`](#impl-Rem%3C%26%27_%20BigInt%3CD2%3E%3E).
    ///
    /// ## Safety
    /// Assuming that `self` and `a` have [sizes][size] `n` and `m` respectively:
    /// * `n >= m > 0`
    /// * Size of `scratch_space` must be exactly
    /// [`ScratchSpace::division_remainder(n, m)`](ScratchSpace::division_remainder) limbs
    ///
    /// Note: we check that these requirements are met using debug assertions.
    ///
    /// [size]: BigInt::size
    ///
    /// ## Constant time
    /// This function is constant time in size of its arguments, ie. number of instructions and
    /// memory access patterns remain the same for the same `n` and `m`.
    pub unsafe fn rem_unchecked(
        &mut self,
        a: &BigInt<impl AsRef<[Digit]>>,
        mut scratch_space: ScratchSpace,
    ) {
        ops::mod_n_m(self.as_mut(), a.as_ref(), scratch_space.as_mut());
    }
}

impl<D: AsRef<[Digit]>> Rem<Digit> for &BigInt<D> {
    type Output = BigInt;

    /// Computes `self mod rhs` where `rhs` is a `Digit`.
    ///
    /// ## Constant time
    /// Irrespective of it's inputs, this method takes `O(self.size())` steps.
    ///
    /// ## Zeroizing
    /// This function might allocate some auxiliary buffer on the heap.
    /// Before deallocating, it gets zeroed out.
    ///
    /// ## Complexity
    /// The complexity is `O(self.size())`.
    ///
    /// ## Panics
    /// The method panics if `rhs` equals zero.
    fn rem(self, rhs: Digit) -> Self::Output {
        self % &BigInt::from_digits(&[rhs])
    }
}

impl<D: AsMut<[Digit]>> BigInt<D> {
    /// Computes `self mod a` where `a` is a digit, writes the result to
    /// the most significant digit of `self`. Accordingly, self must be mutable.
    ///
    /// It is a low-level function allowing you to manage allocations. Its safe analogue is
    /// [`&a % b`](#impl-Rem%3CDigit%3E).
    ///
    /// ## Safety
    /// Assuming that `a` has [size] `n`:
    /// * `n > 0`
    /// * Size of `scratch_space` must be exactly
    /// [`ScratchSpace::division_remainder(n, 1)`](ScratchSpace::division_remainder) limbs
    ///
    /// [size]: BigInt::size
    ///
    /// ## Constant time
    /// This function is constant time in size of its arguments, ie. number of instructions and
    /// memory access patterns remain the same for the same `n`.
    pub unsafe fn rem_digit_unchecked(&mut self, a: Digit, mut scratch_space: ScratchSpace) {
        self.rem_unchecked(&BigInt::from_digits(&[a]), scratch_space);
    }
}

impl_digit_operation!(Rem, rem, u8);
impl_digit_operation_reverse!(Rem, rem, u8);
impl_digit_operation!(Rem, rem, u16);
impl_digit_operation_reverse!(Rem, rem, u16);
impl_digit_operation!(Rem, rem, u32);
impl_digit_operation_reverse!(Rem, rem, u32);
impl_u64_operation!(Rem, rem);
impl_digit_operation_reverse!(Rem, rem, u64);
impl_digit_operation_reverse!(Rem, rem, Digit);

impl<D: AsRef<[Digit]>> AsRef<[Digit]> for BigInt<D> {
    #[inline(always)]
    fn as_ref(&self) -> &[Digit] {
        self.digits.as_ref()
    }
}

impl<D: AsMut<[Digit]>> AsMut<[Digit]> for BigInt<D> {
    #[inline(always)]
    fn as_mut(&mut self) -> &mut [Digit] {
        self.digits.as_mut()
    }
}

/// Auxiliary buffer required for some arithmetic operations
pub struct ScratchSpace<'b>(&'b mut [Digit]);

impl<'b> ScratchSpace<'b> {
    #[inline(always)]
    pub fn new(buffer: &'b mut [Digit]) -> Self {
        Self(buffer)
    }

    /// Size of scratch space required for addition
    #[inline(always)]
    pub fn addition(n: usize, m: usize) -> usize {
        match Ord::cmp(&n, &m) {
            Ordering::Greater => unsafe {
                // Safety: we checked that n > m
                ops::add_n_m_itch(n, m)
            },
            Ordering::Equal => 0,
            Ordering::Less => unsafe {
                // Safety: we checked that m > n
                ops::add_n_m_itch(m, n)
            },
        }
    }

    /// Scratch space size required for digit addition
    #[inline(always)]
    pub fn digit_addition(n: usize) -> usize {
        ops::add_1_itch(n)
    }

    /// Scratch space size required for multiplication
    #[inline(always)]
    pub fn multiplication(n: usize, m: usize) -> usize {
        match Ord::cmp(&n, &m) {
            Ordering::Greater => unsafe {
                // Safety: we checked that n > m
                ops::mul_n_m_itch(n, m)
            },
            Ordering::Equal => 0,
            Ordering::Less => unsafe {
                // Safety: we checked that m > n
                ops::mul_n_m_itch(m, n)
            },
        }
    }

    /// Scratch space size required for division
    #[inline(always)]
    pub fn division(n: usize, m: usize) -> usize {
        match Ord::cmp(&n, &m) {
            Ordering::Greater | Ordering::Equal => unsafe {
                // Safety: we checked that n >= m
                ops::div_n_m_itch(n, m)
            },
            Ordering::Less => 0,
        }
    }

    /// Scratch space size required for division with remainder
    #[inline(always)]
    pub fn division_remainder(n: usize, m: usize) -> usize {
        match Ord::cmp(&n, &m) {
            Ordering::Greater | Ordering::Equal => unsafe {
                // Safety: we checked that n >= m
                ops::mod_n_m_itch(n, m)
            },
            Ordering::Less => 0,
        }
    }
}

impl<'b> From<&'b mut [Digit]> for ScratchSpace<'b> {
    #[inline(always)]
    fn from(buffer: &'b mut [Digit]) -> Self {
        Self(buffer)
    }
}

impl<'b> AsMut<[Digit]> for ScratchSpace<'b> {
    #[inline(always)]
    fn as_mut(&mut self) -> &mut [Digit] {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use crate::digits::tests::strip_padding_u32;
    use crate::BigInt;

    use crate::digits::Digit;
    use std::ops::{Add, Div, Mul, Rem, Sub};

    macro_rules! two_numbers_prop {
        ($op:ident,$a:expr,$b:expr) => {
            let a_gmp = BigInt::from_digits_u32($a);
            let b_gmp = BigInt::from_digits_u32($b);
            let result_gmp = &a_gmp.$op(&b_gmp);
            let limbs_gmp = result_gmp.to_digits_u32();

            let a_num = num_bigint::BigUint::new($a.to_vec());
            let b_num = num_bigint::BigUint::new($b.to_vec());
            let result_num = a_num.$op(b_num);
            let limbs_num = result_num.to_u32_digits();

            prop_assert_eq!(strip_padding_u32(&limbs_gmp), limbs_num.clone());
        };
    }

    macro_rules! number_digit_prop {
        ($op:ident,$a:expr,$b:expr) => {
            let a_gmp = BigInt::from_digits_u32($a).reallocate();
            let result_gmp = &a_gmp.$op($b);
            let limbs_gmp = result_gmp.to_digits_u32();

            let a_num = num_bigint::BigUint::new($a.to_vec());
            let result_num = &a_num.clone().$op($b);
            let limbs_num = result_num.to_u32_digits();

            prop_assert_eq!(strip_padding_u32(&limbs_gmp), limbs_num);
        };
    }

    macro_rules! digit_number_prop {
        ($op:ident,$b:expr,$a:expr) => {
            let a_gmp = BigInt::from_digits_u32($a).reallocate();
            let result_gmp = $b.$op(&a_gmp);
            let limbs_gmp = result_gmp.to_digits_u32();

            let a_num = num_bigint::BigUint::new($a.to_vec());
            let result_num = $b.$op(&a_num);
            let limbs_num = result_num.to_u32_digits();

            prop_assert_eq!(strip_padding_u32(&limbs_gmp), limbs_num);
        };
    }

    proptest! {
        #[test]
        fn add_two_numbers(a: Vec<u32>, b: Vec<u32>) {
            two_numbers_prop!(add, &a, &b);
        }

        #[test]
        fn add_number_and_u8(a: Vec<u32>, b: u8) {
            number_digit_prop!(add, &a, b);
            digit_number_prop!(add, b, &a);
        }

        #[test]
        fn add_number_and_u16(a: Vec<u32>, b: u16) {
            number_digit_prop!(add, &a, b);
            digit_number_prop!(add, b, &a);
        }

        #[test]
        fn add_number_and_u32(a: Vec<u32>, b: u32) {
            number_digit_prop!(add, &a, b);
            digit_number_prop!(add, b, &a);
        }

        #[test]
        fn add_number_and_u64(a: Vec<u32>, b: u64) {
            number_digit_prop!(add, &a, b);
            digit_number_prop!(add, b, &a);
        }

        #[test]
        fn mul_two_numbers(a: Vec<u32>, b: Vec<u32>) {
            two_numbers_prop!(mul, &a, &b);
        }

        #[test]
        fn mul_number_and_u8(a: Vec<u32>, b: u8) {
            number_digit_prop!(mul, &a, b);
            digit_number_prop!(mul, b, &a);
        }

        #[test]
        fn mul_number_and_u16(a: Vec<u32>, b: u16) {
            number_digit_prop!(mul, &a, b);
            digit_number_prop!(mul, b, &a);
        }

        #[test]
        fn mul_number_and_u32(a: Vec<u32>, b: u32) {
            number_digit_prop!(mul, &a, b);
            digit_number_prop!(mul, b, &a);
        }

        #[test]
        fn mul_number_and_u64(a: Vec<u32>, b: u64) {
            number_digit_prop!(mul, &a, b);
            digit_number_prop!(mul, b, &a);
        }

        #[test]
        fn div_two_numbers(a: Vec<u32>, b: Vec<u32>) {
            // divisor must not be zero, otherwise division will rightfully panic
            prop_assume!(b.clone().into_iter().any(|d| d != 0));
            two_numbers_prop!(div, &a, &b);
        }

        #[test]
        fn div_number_and_u8(a: Vec<u32>, b: u8) {
            prop_assume!(b != 0);
            number_digit_prop!(div, &a, b);
        }

        #[test]
        fn div_number_and_u16(a: Vec<u32>, b: u16) {
            prop_assume!(b != 0);
            number_digit_prop!(div, &a, b);
        }

        #[test]
        fn div_number_and_u32(a: Vec<u32>, b: u32) {
            prop_assume!(b != 0);
            number_digit_prop!(div, &a, b);
        }

        #[test]
        fn div_number_and_u64(a: Vec<u32>, b: u64) {
            prop_assume!(b != 0);
            number_digit_prop!(div, &a, b);
        }

        #[test]
        fn rem_two_numbers(a: Vec<u32>, b: Vec<u32>) {
            // divisor must not be zero, otherwise division will rightfully panic
            prop_assume!(b.clone().into_iter().any(|d| d != 0));
            two_numbers_prop!(rem, &a, &b);
        }

        #[test]
        fn rem_number_and_u8(a: Vec<u32>, b: u8) {
            prop_assume!(b != 0);
            number_digit_prop!(rem, &a, b);
        }

        #[test]
        fn rem_number_and_u16(a: Vec<u32>, b: u16) {
            prop_assume!(b != 0);
            number_digit_prop!(rem, &a, b);
        }

        #[test]
        fn rem_number_and_u32(a: Vec<u32>, b: u32) {
            prop_assume!(b != 0);
            number_digit_prop!(rem, &a, b);
        }

        #[test]
        fn rem_number_and_u64(a: Vec<u32>, b: u64) {
            prop_assume!(b != 0);
            number_digit_prop!(rem, &a, b);
        }

        #[test]
        fn division_with_most_significant_zeroes(a: Vec<u32>, b: Vec<u32>) {
            let mut b_mut = b.clone();
            prop_assume!(b.into_iter().any(|d| d != 0));
            let a = &BigInt::from_digits_u32(&a).reallocate();
            // making sure that division and remainder work with zero-padded divisors
            b_mut.extend(&[0]);
            let b = &BigInt::from_digits_u32(&b_mut).reallocate();
            let q = a / b;
            let r = a % b;

            let lhs = &(&q * b) + &r;
            let lhs = lhs.to_digits_u32();
            let rhs = a.to_digits_u32();
            prop_assert_eq!(strip_padding_u32(&lhs), strip_padding_u32(&rhs));
        }
    }

    #[test]
    fn dummy_test() -> Result<(), TestCaseError> {
        two_numbers_prop!(add, &[0], &[0, 0, 0]);
        Ok(())
    }

    #[test]
    #[should_panic(expected = "Division by zero")]
    fn division_by_zero() -> () {
        let a = BigInt::from_digits_u32(&[42, 42]).reallocate();
        let b = BigInt::from_digits_u32(&[0, 0, 0]).reallocate();
        let _ = &a / &b;
        ()
    }
}
