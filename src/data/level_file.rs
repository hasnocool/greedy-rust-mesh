use anyhow::{Context, Result};
use bytemuck::{Pod, Zeroable};
use std::{fs, path::Path};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct ChunkTableEntry {
    pub key: u32,
    pub rle_data_begin: u32,
    pub rle_data_size: u32,
}

#[derive(Debug, Default)]
pub struct LevelFile {
    pub chunk_table: Vec<ChunkTableEntry>,
    pub buffer: Vec<u8>,
    size: u8,
}

impl LevelFile {
    pub fn size(&self) -> u8 {
        self.size
    }

    pub fn load_from_file<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let bytes = fs::read(&path).with_context(|| format!("Failed to read level file: {}", path.as_ref().display()))?;
        anyhow::ensure!(!bytes.is_empty(), "Level file is empty");

        self.size = bytes[0];
        let table_len = (self.size as usize) * (self.size as usize);
        let table_bytes = table_len * std::mem::size_of::<ChunkTableEntry>();
        anyhow::ensure!(bytes.len() >= 1 + table_bytes, "Level file is truncated (missing chunk table)");

        let table_start = 1;
        let table_end = table_start + table_bytes;
        let table_slice = &bytes[table_start..table_end];
        // The on-disk chunk table is tightly packed bytes; it may not be aligned for safe casting.
        // Decode manually as little-endian u32 triplets.
        self.chunk_table.clear();
        self.chunk_table.reserve(table_len);
        for i in 0..table_len {
            let base = i * 12;
            let key = u32::from_le_bytes(table_slice[base..base + 4].try_into().unwrap());
            let rle_data_begin = u32::from_le_bytes(table_slice[base + 4..base + 8].try_into().unwrap());
            let rle_data_size = u32::from_le_bytes(table_slice[base + 8..base + 12].try_into().unwrap());
            self.chunk_table.push(ChunkTableEntry {
                key,
                rle_data_begin,
                rle_data_size,
            });
        }
        self.buffer = bytes;
        Ok(())
    }
}
