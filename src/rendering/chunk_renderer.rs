use crate::CS;
use crate::mesher::QuadData;
use anyhow::{anyhow, Result};
use bytemuck::{Pod, Zeroable};
use glow::HasContext;
use std::rc::Rc;

pub const BUFFER_SIZE_BYTES: usize = 512 * 1024 * 1024; // 512MB
pub const QUAD_SIZE_BYTES: usize = 8;

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
pub struct DrawElementsIndirectCommand {
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub base_vertex: u32,
    pub base_instance: u32,
}

pub struct ChunkRenderer {
    gl: Rc<glow::Context>,
    vao: glow::NativeVertexArray,
    ibo: glow::NativeBuffer,
    ssbo: glow::NativeBuffer,
    command_buffer: glow::NativeBuffer,

    pub draw_commands: Vec<DrawElementsIndirectCommand>,
    allocation_end_bytes: usize,
}

impl ChunkRenderer {
    pub fn new(gl: &Rc<glow::Context>) -> Result<Self> {
        unsafe {
            let vao = gl
                .create_vertex_array()
                .map_err(|e| anyhow!("create VAO failed: {e}"))?;
            let ibo = gl.create_buffer().map_err(|e| anyhow!("create IBO failed: {e}"))?;
            let ssbo = gl.create_buffer().map_err(|e| anyhow!("create SSBO failed: {e}"))?;
            let command_buffer = gl
                .create_buffer()
                .map_err(|e| anyhow!("create indirect buffer failed: {e}"))?;

            gl.bind_vertex_array(Some(vao));

            // SSBO
            gl.bind_buffer(glow::SHADER_STORAGE_BUFFER, Some(ssbo));
            gl.buffer_data_size(glow::SHADER_STORAGE_BUFFER, BUFFER_SIZE_BYTES as i32, glow::DYNAMIC_DRAW);
            gl.bind_buffer(glow::SHADER_STORAGE_BUFFER, None);

            // IBO indices (enough for worst-case number of quads in a face: CS^3)
            let max_quads = CS * CS * CS * 6;
            let mut indices: Vec<u32> = Vec::with_capacity(max_quads * 6);
            for i in 0..(max_quads as u32) {
                indices.push((i << 2) | 2);
                indices.push((i << 2) | 0);
                indices.push((i << 2) | 1);
                indices.push((i << 2) | 1);
                indices.push((i << 2) | 3);
                indices.push((i << 2) | 2);
            }

            gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(ibo));
            gl.buffer_data_u8_slice(
                glow::ELEMENT_ARRAY_BUFFER,
                bytemuck::cast_slice(&indices),
                glow::STATIC_DRAW,
            );

            // Indirect command buffer
            gl.bind_buffer(glow::DRAW_INDIRECT_BUFFER, Some(command_buffer));
            gl.buffer_data_size(
                glow::DRAW_INDIRECT_BUFFER,
                (100_000usize * std::mem::size_of::<DrawElementsIndirectCommand>()) as i32,
                glow::DYNAMIC_DRAW,
            );

            gl.bind_vertex_array(None);
            gl.bind_buffer(glow::DRAW_INDIRECT_BUFFER, None);

            Ok(Self {
                gl: Rc::clone(gl),
                vao,
                ibo,
                ssbo,
                command_buffer,
                draw_commands: Vec::new(),
                allocation_end_bytes: 0,
            })
        }
    }

    pub fn upload_quads(&mut self, quads: &[QuadData]) -> Result<u32> {
        // Returns base_vertex (in vertices, i.e. quad_index*4)
        let bytes = quads.len() * QUAD_SIZE_BYTES;
        anyhow::ensure!(self.allocation_end_bytes + bytes <= BUFFER_SIZE_BYTES, "SSBO out of space");

        let base_quad = self.allocation_end_bytes / QUAD_SIZE_BYTES;
        unsafe {
            self.gl.bind_buffer(glow::SHADER_STORAGE_BUFFER, Some(self.ssbo));
            self.gl.buffer_sub_data_u8_slice(
                glow::SHADER_STORAGE_BUFFER,
                self.allocation_end_bytes as i32,
                bytemuck::cast_slice(quads),
            );
            self.gl.bind_buffer(glow::SHADER_STORAGE_BUFFER, None);
        }

        self.allocation_end_bytes += bytes;
        Ok((base_quad as u32) << 2)
    }

    pub fn add_draw_command(&mut self, cmd: DrawElementsIndirectCommand) {
        self.draw_commands.push(cmd);
    }

    pub fn render(&mut self) {
        if self.draw_commands.is_empty() {
            return;
        }

        unsafe {
            self.gl.bind_buffer(glow::DRAW_INDIRECT_BUFFER, Some(self.command_buffer));
            self.gl.buffer_data_u8_slice(
                glow::DRAW_INDIRECT_BUFFER,
                bytemuck::cast_slice(&self.draw_commands),
                glow::DYNAMIC_DRAW,
            );

            self.gl.bind_vertex_array(Some(self.vao));
            self.gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(self.ibo));
            self.gl.bind_buffer_base(glow::SHADER_STORAGE_BUFFER, 0, Some(self.ssbo));

            // Draw one indirect command at a time for broad compatibility.
            for i in 0..self.draw_commands.len() {
                let offset = (i * std::mem::size_of::<DrawElementsIndirectCommand>()) as i32;
                self.gl
                    .draw_elements_indirect_offset(glow::TRIANGLES, glow::UNSIGNED_INT, offset);
            }

            self.gl.bind_buffer_base(glow::SHADER_STORAGE_BUFFER, 0, None);
            self.gl.bind_vertex_array(None);
            self.gl.bind_buffer(glow::DRAW_INDIRECT_BUFFER, None);
        }

        self.draw_commands.clear();
    }
}

impl Drop for ChunkRenderer {
    fn drop(&mut self) {
        unsafe {
            self.gl.delete_buffer(self.ibo);
            self.gl.delete_buffer(self.ssbo);
            self.gl.delete_buffer(self.command_buffer);
            self.gl.delete_vertex_array(self.vao);
        }
    }
}
