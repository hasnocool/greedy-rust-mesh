use crate::{CS_P2};

#[inline]
fn get_bit_range(low: u8, high: u8) -> u64 {
    // inclusive low..=high
    let width = (high - low + 1) as u32;
    if width >= 64 {
        u64::MAX
    } else {
        ((1u64 << width) - 1) << (low as u32)
    }
}

pub fn decompress_to_voxels_and_opaque_mask(rle: &[u8], voxels: &mut [u8], opaque_mask: &mut [u64]) {
    debug_assert_eq!(opaque_mask.len(), CS_P2);

    // opaque_mask is expected to be zeroed by caller
    let mut u_i: usize = 0;

    let mut opaque_mask_index: usize = 0;
    let mut opaque_mask_bit_index: u8 = 0;

    let mut p: usize = 0;
    while p + 1 < rle.len() {
        let ty = rle[p];
        let len = rle[p + 1] as usize;
        p += 2;

        if len == 0 {
            continue;
        }

        voxels[u_i..u_i + len].fill(ty);

        // Decompress into opaque mask (bitstream over the CS_P3 voxel buffer)
        let mut remaining = len;
        while remaining > 0 {
            let remaining_bits_in_index = 64u8.saturating_sub(opaque_mask_bit_index) as usize;

            if remaining < remaining_bits_in_index {
                if ty != 0 {
                    opaque_mask[opaque_mask_index] |= get_bit_range(
                        opaque_mask_bit_index,
                        opaque_mask_bit_index + (remaining as u8) - 1,
                    );
                }
                opaque_mask_bit_index = opaque_mask_bit_index + remaining as u8;
                remaining = 0;
            } else if remaining >= 64 && opaque_mask_bit_index == 0 {
                let count = remaining / 64;
                if ty != 0 {
                    for v in &mut opaque_mask[opaque_mask_index..opaque_mask_index + count] {
                        *v = u64::MAX;
                    }
                }
                opaque_mask_index += count;
                remaining -= count * 64;
            } else {
                if ty != 0 {
                    opaque_mask[opaque_mask_index] |= get_bit_range(opaque_mask_bit_index, 63);
                }
                remaining -= remaining_bits_in_index;
                opaque_mask_index += 1;
                opaque_mask_bit_index = 0;
            }
        }

        u_i += len;
    }
}
