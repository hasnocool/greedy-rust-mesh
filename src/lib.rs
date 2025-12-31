pub mod data;
pub mod mesher;
pub mod misc;
pub mod rendering;

pub const CS: usize = 62;
pub const CS_P: usize = CS + 2;
pub const CS_2: usize = CS * CS;
pub const CS_P2: usize = CS_P * CS_P;
pub const CS_P3: usize = CS_P * CS_P * CS_P;

pub fn get_zxy_index(x: usize, y: usize, z: usize) -> usize {
    z + (x * CS_P) + (y * CS_P2)
}

pub fn get_xyz_key(x: u8, y: u8, z: u8) -> u32 {
    ((z as u32) << 16) | ((y as u32) << 8) | (x as u32)
}

pub fn parse_xyz_key(key: u32) -> (u8, u8, u8) {
    ((key & 0xFF) as u8, ((key >> 8) & 0xFF) as u8, ((key >> 16) & 0xFF) as u8)
}
