//! Convenient utils to work with [Digit]s

use std::borrow::Cow;
use std::hint::unreachable_unchecked;
use std::mem::size_of;
use std::{ops, slice};

use gmp_mpfr_sys::gmp::limb_t;

// We expect that limb (ie Digit) is either 32 or 64 bits wide. Otherwise code in this module is
// incorrect.
static_assertions::const_assert!(size_of::<Digit>() == 4 || size_of::<Digit>() == 8);

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Default)]
#[repr(transparent)]
pub struct Digit(limb_t);

impl Digit {
    #[inline(always)]
    pub fn from_limb(limb: limb_t) -> Self {
        Self(limb)
    }
    #[inline(always)]
    pub fn limb(self) -> limb_t {
        self.0
    }
    #[inline(always)]
    pub fn zero() -> Self {
        Self(0)
    }
}

/// Converts slice of limbs into slice of digits
pub fn digits_from_limbs(limbs: &[limb_t]) -> &[Digit] {
    // Safety: Digit has the same memory representation as limb_t, see
    // `#[repr(transparent)]` attribute
    unsafe { slice::from_raw_parts(limbs.as_ptr() as _, limbs.len()) }
}

/// Converts u32 limbs into digits
pub fn digits_from_limbs_u32(limbs: &[u32]) -> Cow<[Digit]> {
    match size_of::<Digit>() {
        4 => {
            let slice = unsafe {
                // Safety: Digit has the same memory representation as u32 in this branch
                slice::from_raw_parts(limbs.as_ptr() as *const Digit, limbs.len())
            };
            Cow::Borrowed(slice)
        }
        8 => {
            let digits: Vec<_> = limbs
                .chunks(2)
                .map(|chunk| {
                    let mut digit = chunk[0] as u64;
                    if let Some(&hi) = chunk.get(1) {
                        digit |= (hi as u64) << 32
                    }
                    digit
                })
                .collect();
            let digits = unsafe {
                let digits = std::mem::ManuallyDrop::new(digits);
                // Safety: Digit has the same memory representation as u64 in this branch
                Vec::from_raw_parts(
                    digits.as_ptr() as *mut limb_t as *mut Digit,
                    digits.len(),
                    digits.capacity(),
                )
            };
            Cow::Owned(digits)
        }
        _ => {
            // SAFETY: we statically assert that Digit size is either 4 or 8 (see const_assert!
            // macro call)
            unsafe { unreachable_unchecked() }
        }
    }
}

/// Converts digits to u32 limbs
pub fn digits_to_limbs_u32(digits: &[Digit]) -> Cow<[u32]> {
    match size_of::<Digit>() {
        4 => {
            let slice = unsafe {
                // Safety: Digit has the same memory representation as u32 in this branch
                slice::from_raw_parts(digits.as_ptr() as *const u32, digits.len())
            };
            Cow::Borrowed(slice)
        }
        8 => {
            let digits = unsafe {
                // Safety: Digit has the same memory representation as u64 in this branch
                slice::from_raw_parts(digits.as_ptr() as *const u64, digits.len())
            };

            let limbs: Vec<_> = digits
                .iter()
                .flat_map(|&digit| [(digit & 0xFFFF_FFFF) as u32, (digit >> 32) as u32])
                .collect();

            Cow::Owned(limbs)
        }
        _ => {
            // SAFETY: we statically assert that Digit size is either 4 or 8 (see const_assert!
            // macro call)
            unsafe { unreachable_unchecked() }
        }
    }
}

impl From<u32> for Digit {
    fn from(i: u32) -> Digit {
        Digit::from_limb(i.into())
    }
}

impl From<u16> for Digit {
    fn from(i: u16) -> Digit {
        Digit::from_limb(i.into())
    }
}

impl From<u8> for Digit {
    fn from(i: u8) -> Digit {
        Digit::from_limb(i.into())
    }
}

impl From<bool> for Digit {
    fn from(i: bool) -> Digit {
        Digit::from_limb(i.into())
    }
}

impl zeroize::DefaultIsZeroes for Digit {}

/// [u128] integer represented as 2 or 4 [Digit]s
pub struct U128([Digit; U128::size_in_limbs()]);

impl U128 {
    pub const fn size_in_limbs() -> usize {
        size_of::<u128>() / size_of::<Digit>()
    }
}

impl From<u128> for U128 {
    fn from(i: u128) -> U128 {
        const U32_MASK: u128 = 0xFFFF_FFFF;
        const U64_MASK: u128 = 0xFFFF_FFFF_FFFF_FFFF;

        let mut arr = [Digit::from_limb(0); U128::size_in_limbs()];

        #[allow(unconditional_panic)]
        match size_of::<Digit>() {
            4 => {
                arr[0] = Digit::from_limb((i & U32_MASK) as _);
                arr[1] = Digit::from_limb(((i >> 32) & U32_MASK) as _);
                arr[2] = Digit::from_limb(((i >> 64) & U32_MASK) as _);
                arr[3] = Digit::from_limb((i >> 96) as _);
            }
            8 => {
                arr[0] = Digit::from_limb((i & U64_MASK) as _);
                arr[1] = Digit::from_limb((i >> 64) as _)
            }
            _ => {
                // SAFETY: we statically assert that Digit size is either 4 or 8 (see const_assert!
                // macro call)
                unsafe { unreachable_unchecked() }
            }
        }
        U128(arr)
    }
}

impl ops::Deref for U128 {
    type Target = [Digit];
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl ops::DerefMut for U128 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl AsRef<[Digit]> for U128 {
    fn as_ref(&self) -> &[Digit] {
        &self.0
    }
}

impl AsMut<[Digit]> for U128 {
    fn as_mut(&mut self) -> &mut [Digit] {
        &mut self.0
    }
}

/// [u64] integer represented as 1 or 2 [Digit]s
pub struct U64([Digit; U64::size_in_limbs()]);

impl U64 {
    pub const fn size_in_limbs() -> usize {
        size_of::<u64>() / size_of::<Digit>()
    }
}

impl From<u64> for U64 {
    fn from(i: u64) -> Self {
        const U32_MASK: u64 = 0xFFFF_FFFF;

        let mut arr = [Digit::from_limb(0); U64::size_in_limbs()];

        #[allow(unconditional_panic)]
        match size_of::<Digit>() {
            4 => {
                arr[0] = Digit::from_limb((i & U32_MASK) as _);
                arr[1] = Digit::from_limb((i >> 32) as _);
            }
            8 => {
                arr[0] = Digit::from_limb(i as _);
            }
            _ => {
                // SAFETY: we statically assert that Digit size is either 4 or 8 (see const_assert!
                // macro call)
                unsafe { unreachable_unchecked() }
            }
        }

        U64(arr)
    }
}

impl ops::Deref for U64 {
    type Target = [Digit];
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl ops::DerefMut for U64 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl AsRef<[Digit]> for U64 {
    fn as_ref(&self) -> &[Digit] {
        &self.0
    }
}

impl AsMut<[Digit]> for U64 {
    fn as_mut(&mut self) -> &mut [Digit] {
        &mut self.0
    }
}

#[cfg(test)]
pub mod tests {
    use std::convert::TryFrom;

    use proptest::prelude::*;

    use super::*;

    proptest! {
        #[test]
        fn u128_to_limbs(i: u128) {
            u128_to_limbs_prop(i)?;
        }
        #[test]
        fn u64_to_limbs(i: u64) {
            u64_to_limbs_prop(i)?;
        }
        #[test]
        fn u32_to_limb(i: u32) {
            u32_to_limb_prop(i)?;
        }
        #[test]
        fn u16_to_limb(i: u16) {
            u16_to_limb_prop(i)?;
        }
        #[test]
        fn u8_to_limb(i: u8) {
            u8_to_limb_prop(i)?;
        }
        #[test]
        fn bool_to_limb(i: bool) {
            bool_to_limb_prop(i)?;
        }
        #[test]
        fn five_u32_limbs_to_digits(
            a: u32,
            b: u32,
            c: u32,
            d: u32,
            e: u32,
        ) {
            five_u32_limbs_to_digits_prop(a,b,c,d,e)?
        }
        #[test]
        fn u32_limbs_to_digits_and_backwards(limbs: Vec<u32>) {
            u32_limbs_to_digits_and_backwards_prop(&limbs)?
        }
    }

    fn u128_to_limbs_prop(i: u128) -> Result<(), TestCaseError> {
        let limbs = U128::from(i);

        let mut reconstructed_i = 0u128;
        for &limb in (&*limbs).iter().rev() {
            reconstructed_i <<= size_of::<Digit>() * 8;
            reconstructed_i |= limb.0 as u128;
        }
        prop_assert_eq!(i, reconstructed_i);

        Ok(())
    }

    fn u64_to_limbs_prop(i: u64) -> Result<(), TestCaseError> {
        let limbs = U64::from(i);

        let mut reconstructed_i = 0u64;
        match size_of::<Digit>() {
            4 => {
                for &limb in (&*limbs).iter().rev() {
                    reconstructed_i <<= size_of::<Digit>() * 8;
                    reconstructed_i |= limb.limb() as u64;
                }
            }
            8 => {
                reconstructed_i = limbs[0].limb() as _;
            }
            _ => unreachable!(),
        }

        prop_assert_eq!(i, reconstructed_i);

        Ok(())
    }

    fn u32_to_limb_prop(i: u32) -> Result<(), TestCaseError> {
        let limb = Digit::from(i).limb();
        prop_assert_eq!(limb, i.into());
        Ok(())
    }

    fn u16_to_limb_prop(i: u16) -> Result<(), TestCaseError> {
        let limb = Digit::from(i).limb();
        prop_assert_eq!(u16::try_from(limb).unwrap(), i);
        Ok(())
    }

    fn u8_to_limb_prop(i: u8) -> Result<(), TestCaseError> {
        let limb = Digit::from(i).limb();
        prop_assert_eq!(u8::try_from(limb).unwrap(), i);
        Ok(())
    }

    fn bool_to_limb_prop(i: bool) -> Result<(), TestCaseError> {
        let limb = Digit::from(i).limb();
        let reconstructed_bool = match limb {
            0 => false,
            1 => true,
            _ => panic!(),
        };
        prop_assert_eq!(reconstructed_bool, i);
        Ok(())
    }

    fn five_u32_limbs_to_digits_prop(
        a: u32,
        b: u32,
        c: u32,
        d: u32,
        e: u32,
    ) -> Result<(), TestCaseError> {
        let limbs = &[a, b, c, d, e];
        let digits = digits_from_limbs_u32(limbs);

        let l1 = U64::from(((b as u64) << 32) | (a as u64));
        let l2 = U64::from(((d as u64) << 32) | (c as u64));
        let l3 = U64::from(e as u64);

        let digits_expected: Vec<_> = [l1, l2, l3]
            .iter()
            .flat_map(|l| l.as_ref())
            .copied()
            .collect();

        match size_of::<Digit>() {
            4 => prop_assert_eq!(digits, &digits_expected[..digits_expected.len() - 1]),
            8 => prop_assert_eq!(digits, digits_expected),
            _ => unreachable!(),
        }

        Ok(())
    }

    fn u32_limbs_to_digits_and_backwards_prop(limbs: &[u32]) -> Result<(), TestCaseError> {
        let digits = digits_from_limbs_u32(limbs);
        let limbs2 = digits_to_limbs_u32(&digits);
        prop_assert_eq!(strip_padding_u32(limbs), strip_padding_u32(&limbs2));
        Ok(())
    }

    pub fn strip_padding_u32(limbs: &[u32]) -> &[u32] {
        let index = limbs
            .iter()
            .enumerate()
            .rev()
            .find(|(_, &limb)| limb != 0)
            .map(|(ind, _)| ind);
        match index {
            Some(index) => limbs.split_at(index + 1).0,
            None => &[],
        }
    }
}
