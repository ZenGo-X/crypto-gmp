use std::borrow::Cow;
use std::cmp::Ordering;

use gmp_mpfr_sys::gmp::size_t;
use subtle::{Choice, ConditionallySelectable};
use zeroize::Zeroize;

use crate::digits::Digit;

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

impl<D1, D2> std::ops::Add<&BigInt<D2>> for &BigInt<D1>
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
    /// Addition complexity is `O(max(self.size(), rhs.size())`
    fn add(self, rhs: &BigInt<D2>) -> Self::Output {
        // We have to maintain a decreasing order of the input arguments:
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

        // Estimate the space we need to allocate on a heap
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
    /// It is a low-level function allowing you to manage allocations. Its safe analogous is
    /// [`&a + &b`](#impl-Add%3C%26%27_%20BigInt%3CD2%3E%3E)
    ///
    /// ## Safety
    /// Assuming that `a` and `b` have [sizes][size] `n` and `m` respectively:
    /// * `n >= m > 0`
    /// * [Size][size] of `self` must be exactly `n+1` bytes
    /// * Size of `scratch_space` must be exactly [`ScratchSpace::addition(n, m)`](ScratchSpace::addition)
    ///
    /// Note: we check that these requirements meet using debug assertions.
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
            self.as_mut()[a.size()] = carry;
        }
    }
}

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
        &mut self.0
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use crate::digits::tests::strip_padding_u32;
    use crate::BigInt;

    proptest! {
        #[test]
        fn add_two_numbers(a: Vec<u32>, b: Vec<u32>) {
            add_two_numbers_prop(&a, &b)?
        }
    }

    #[test]
    fn dummy_test() -> Result<(), TestCaseError> {
        add_two_numbers_prop(&[0], &[0, 0, 0])
    }

    fn add_two_numbers_prop(a: &[u32], b: &[u32]) -> Result<(), TestCaseError> {
        // We assume that a and b are not negative
        prop_assume!(a.is_empty() || (a.last().unwrap() >> 31) == 0);
        prop_assume!(b.is_empty() || (b.last().unwrap() >> 31) == 0);

        let a_gmp = BigInt::from_digits_u32(a);
        let b_gmp = BigInt::from_digits_u32(b);
        let result_gmp = &a_gmp + &b_gmp;
        let limbs_gmp = result_gmp.to_digits_u32();

        let a_num = num_bigint::BigUint::new(a.to_vec());
        let b_num = num_bigint::BigUint::new(b.to_vec());
        let result_num = a_num + b_num;
        let limbs_num = result_num.to_u32_digits();

        prop_assert_eq!(strip_padding_u32(&limbs_gmp), limbs_num);

        Ok(())
    }
}
