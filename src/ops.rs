use std::slice;

use gmp_mpfr_sys::gmp::{self, limb_t, size_t};
use subtle::{Choice, ConditionallySelectable};

use crate::digits::Digit;

/// Convenient wrapper to [mpn_add_n](gmp::mpn_add_n)
///
/// # Safety
/// * `a`, `b`, `output` must be of the same size, they must not be empty
pub unsafe fn add_n(a: &[Digit], b: &[Digit], output: &mut [Digit]) -> Digit {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), output.len());
    debug_assert!(!a.is_empty());

    let n = a.len() as size_t;
    let a = a.as_ptr() as *const limb_t;
    let b = b.as_ptr() as *const limb_t;
    let output = output.as_mut_ptr() as *mut limb_t;

    let carry_limb = gmp::mpn_add_n(output, a, b, n);
    Digit::from_limb(carry_limb)
}

/// Convenient wrapper to [mpn_sec_add_1](gmp::mpn_sec_add_1)
///
/// # Safety
/// * `a`, `output` must be of the same size `n`
/// * `scratch_space` must be of size `add_1_itch(n)`
/// * `n != 0`
pub unsafe fn add_1(
    a: &[Digit],
    b: Digit,
    scratch_space: &mut [Digit],
    output: &mut [Digit],
) -> Digit {
    debug_assert_eq!(a.len(), output.len());
    debug_assert!(!a.is_empty());

    let n = a.len() as size_t;
    debug_assert_eq!(scratch_space.len() as size_t, gmp::mpn_sec_add_1_itch(n));

    let a = a.as_ptr() as *const limb_t;
    let scratch_space = scratch_space.as_mut_ptr() as *mut limb_t;
    let output = output.as_mut_ptr() as *mut limb_t;

    let carry_limb = gmp::mpn_sec_add_1(output, a, n, b.limb(), scratch_space);
    Digit::from_limb(carry_limb)
}

pub fn add_1_itch(n: usize) -> usize {
    unsafe { gmp::mpn_sec_add_1_itch(n as size_t) as usize }
}

/// Takes two integers `a` and `b` represented with sequence of limbs of size `n` and `m` in order
/// from least to most significant. Computes addition `a+b` and writes it to `output`. Return a carry
/// digit.
///
/// ## Safety
/// Following requirements must be met:
/// * `n > m > 0`
///   Ie. size of `a` must be greater than `b`. It does not implies that `a > b` since both numbers
///   can be padded with zeroes. `b` must not be empty.
/// * Size of `output` must be exactly `n` bytes. `output` may be filled with arbitrary data.
/// * Size of `scratch_space` must be exactly `add_n_m_itch(n, m)` bytes. `scratch_space` may be
///   filled with arbitrary data.
///
/// ## Guarantees
/// Function is guaranteed to be constant time in size of its arguments. Ie. amount of instructions
/// and memory access patterns should be the same for all numbers with the same `n` and `m`.
///
/// ## Algorithm
/// Implementation is based on [mpn_add_n] and [mpn_sec_add_1] functions.
///
/// Recall that as input we have two differently sized numbers `a` and `b` represented as sequence of
/// limbs in order from least to most significant. E.g.:
///
/// ```text
/// a = [1122, 2233, 3344, 4455] n=4
/// b = [3322, 2211]             m=2
/// ```
///
/// 1) We take `m` least significant limbs from each number (ie. `&a[0..m]` and `&b[0..m]`) and
///    add them using [mpn_add_n] (which is constant time). Result is written to `&mut output[0..m]`.
///    Function returns a `carry` limb
/// 2) We take `n-m` most significant limbs from number `a` (`&a[m..n-m]`) and add to them a `carry`
///    limb using [mpn_sec_add_1]. Result is written to `&mut output[m..n]`. Function return a
///    `carry2` limb
/// 3) We return `carry2`
///
/// [mpn_add_n]: gmp::mpn_add_n
/// [mpn_sec_add_1]: gmp::mpn_sec_add_1
pub unsafe fn add_n_m(
    a: &[Digit],
    b: &[Digit],
    scratch_space: &mut [Digit],
    output: &mut [Digit],
) -> Digit {
    debug_assert!(a.len() > b.len());
    debug_assert!(!b.is_empty());
    debug_assert_eq!(output.len(), a.len());
    debug_assert_eq!(scratch_space.len(), add_n_m_itch(a.len(), b.len()));

    let n = a.len() as size_t;
    let m = b.len() as size_t;
    let pa = a.as_ptr() as *const limb_t;
    let pb = b.as_ptr() as *const limb_t;
    let pout = output.as_mut_ptr() as *mut limb_t;
    let scratch_space = scratch_space.as_mut_ptr() as *mut limb_t;

    let carry = gmp::mpn_add_n(pout, pa, pb, m);

    let pa = pa.add(b.len());
    let output = pout.add(b.len());
    let tail = n - m;
    let carry = gmp::mpn_sec_add_1(output, pa, tail, carry, scratch_space);

    Digit::from_limb(carry)
}

/// Returns size of scratch space required for [add_n_m] function.
///
/// ## Safety
/// Requires that `n` is greater than `m`, ie `n > m`
pub unsafe fn add_n_m_itch(n: usize, m: usize) -> usize {
    gmp::mpn_sec_add_1_itch((n - m) as _) as _
}

/// Takes two integers `a` and `b` represented with sequence of limbs of size `n` and `m` in order
/// from least to most significant
///
/// Returns:
/// * `(a, b)` if `n > m`
/// * `(b, a)` otherwise
///
/// Function is constant time.
pub fn reorder<'n>(a: &'n [Digit], b: &'n [Digit]) -> (&'n [Digit], &'n [Digit]) {
    let mut n = a.len() as size_t;
    let mut m = b.len() as size_t;

    // Since size_t is machine word, comparison should take one instruction
    let n_is_greater = Choice::from(u8::from(n > m));

    let mut a_ptr = a.as_ptr() as size_t;
    let mut b_ptr = b.as_ptr() as size_t;

    size_t::conditional_swap(&mut a_ptr, &mut b_ptr, !n_is_greater);
    size_t::conditional_swap(&mut n, &mut m, !n_is_greater);

    let a = unsafe { slice::from_raw_parts(a_ptr as _, n as _) };
    let b = unsafe { slice::from_raw_parts(b_ptr as _, m as _) };

    (a, b)
}

#[cfg(test)]
mod tests {
    use proptest::test_runner::TestCaseError;
    use proptest::{prop_assert_eq, prop_assume, proptest};

    use crate::digits::{digits_from_limbs, U128};

    use super::*;

    proptest! {
        #[test]
        fn add_n_u128(a: u128, b: u128) {
            add_n_u128_prop(a, b)?
        }
        #[test]
        fn add_1_u128_u32(a: u128, b: u32) {
            add_1_u128_u32_prop(a, b)?
        }
        #[test]
        fn add_n_m_u256_u128(a_hi: u128, a_lo: u128, b: u128) {
            add_n_m_u256_u128_prop(a_hi, a_lo, b)?
        }
        #[test]
        fn reorder_two_numbers(a: Vec<limb_t>, b: Vec<limb_t>) {
            reorder_prop(digits_from_limbs(&a), digits_from_limbs(&b))?
        }
    }

    fn add_n_u128_prop(a: u128, b: u128) -> Result<(), TestCaseError> {
        let a_bytes = U128::from(a);
        let b_bytes = U128::from(b);
        let mut output = U128::from(0);

        let carry_limb = unsafe { add_n(&*a_bytes, &*b_bytes, &mut *output) };

        let (output_expected, carry_expected) = u128::overflowing_add(a, b);
        prop_assert_eq!(&*output, &*U128::from(output_expected));
        prop_assert_eq!(carry_limb, Digit::from(carry_expected));

        Ok(())
    }

    fn add_1_u128_u32_prop(n: u128, m: u32) -> Result<(), TestCaseError> {
        let n_digits = U128::from(n);
        let m_digit = Digit::from(m);
        let mut output = U128::from(0);
        let mut scratch_space = vec![Digit::zero(); add_1_itch(n_digits.len())];

        let carry = unsafe { add_1(&n_digits, m_digit, &mut scratch_space, &mut output) };

        let (output_expected, carry_expected) = u128::overflowing_add(n, m.into());
        prop_assert_eq!(&*output, &*U128::from(output_expected));
        prop_assert_eq!(carry, Digit::from(carry_expected));

        Ok(())
    }

    fn add_n_m_u256_u128_prop(a_hi: u128, a_lo: u128, b: u128) -> Result<(), TestCaseError> {
        let mut a_limbs = [Digit::zero(); 2 * U128::size_in_limbs()];
        a_limbs[0..U128::size_in_limbs()].copy_from_slice(&U128::from(a_lo));
        a_limbs[U128::size_in_limbs()..].copy_from_slice(&U128::from(a_hi));

        let b_limbs = U128::from(b);
        let mut output = [Digit::zero(); 2 * U128::size_in_limbs()];
        let scratch_space_size =
            unsafe { add_n_m_itch(2 * U128::size_in_limbs(), U128::size_in_limbs()) };
        let mut scratch_space = vec![Digit::zero(); scratch_space_size];

        let carry = unsafe { add_n_m(&a_limbs, &b_limbs, &mut scratch_space, &mut output) };

        let (output_hi, output_lo, expected_carry) = {
            let (lo, c) = u128::overflowing_add(a_lo, b);
            let (hi, c) = u128::overflowing_add(a_hi, u128::from(c));
            (hi, lo, c)
        };

        prop_assert_eq!(&output[0..U128::size_in_limbs()], &*U128::from(output_lo));
        prop_assert_eq!(&output[U128::size_in_limbs()..], &*U128::from(output_hi));
        prop_assert_eq!(carry, Digit::from(expected_carry));

        Ok(())
    }

    fn reorder_prop(a: &[Digit], b: &[Digit]) -> Result<(), TestCaseError> {
        prop_assume!(a.len() != b.len());

        let (p1, p2) = reorder(a, b);
        let (q1, q2) = reorder(b, a);

        prop_assert_eq!(p1, q1);
        prop_assert_eq!(p2, q2);

        let (ex1, ex2) = if a.len() > b.len() { (a, b) } else { (b, a) };

        prop_assert_eq!(p1, ex1);
        prop_assert_eq!(p2, ex2);

        Ok(())
    }
}
