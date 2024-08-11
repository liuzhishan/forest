
/// Different process method of sign
pub trait SignConverter {
    fn convert(self, sign: u64) -> u64;
}
