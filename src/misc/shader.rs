use anyhow::{anyhow, Result};
use glow::HasContext;
use std::rc::Rc;

pub struct ShaderProgram {
    gl: Rc<glow::Context>,
    pub id: glow::NativeProgram,
}

impl ShaderProgram {
    pub fn new(gl: &Rc<glow::Context>, vertex_src: &str, fragment_src: &str) -> Result<Self> {
        unsafe {
            let program = gl
                .create_program()
                .map_err(|e| anyhow!("create_program failed: {e}"))?;

            let vs = gl
                .create_shader(glow::VERTEX_SHADER)
                .map_err(|e| anyhow!("create vertex shader failed: {e}"))?;
            gl.shader_source(vs, vertex_src);
            gl.compile_shader(vs);
            if !gl.get_shader_compile_status(vs) {
                let log = gl.get_shader_info_log(vs);
                gl.delete_shader(vs);
                gl.delete_program(program);
                return Err(anyhow!("Vertex shader compile failed: {log}"));
            }

            let fs = gl
                .create_shader(glow::FRAGMENT_SHADER)
                .map_err(|e| anyhow!("create fragment shader failed: {e}"))?;
            gl.shader_source(fs, fragment_src);
            gl.compile_shader(fs);
            if !gl.get_shader_compile_status(fs) {
                let log = gl.get_shader_info_log(fs);
                gl.delete_shader(vs);
                gl.delete_shader(fs);
                gl.delete_program(program);
                return Err(anyhow!("Fragment shader compile failed: {log}"));
            }

            gl.attach_shader(program, vs);
            gl.attach_shader(program, fs);
            gl.link_program(program);
            if !gl.get_program_link_status(program) {
                let log = gl.get_program_info_log(program);
                gl.delete_shader(vs);
                gl.delete_shader(fs);
                gl.delete_program(program);
                return Err(anyhow!("Program link failed: {log}"));
            }

            gl.detach_shader(program, vs);
            gl.detach_shader(program, fs);
            gl.delete_shader(vs);
            gl.delete_shader(fs);

            Ok(Self {
                gl: Rc::clone(gl),
                id: program,
            })
        }
    }

    pub fn bind(&self) {
        unsafe {
            self.gl.use_program(Some(self.id));
        }
    }

    pub fn uniform_location(&self, name: &str) -> Option<glow::NativeUniformLocation> {
        unsafe { self.gl.get_uniform_location(self.id, name) }
    }

    pub fn set_mat4(&self, loc: &glow::NativeUniformLocation, mat: &glam::Mat4) {
        unsafe { self.gl.uniform_matrix_4_f32_slice(Some(loc), false, &mat.to_cols_array()) }
    }

    pub fn set_vec3(&self, loc: &glow::NativeUniformLocation, v: &glam::Vec3) {
        unsafe { self.gl.uniform_3_f32(Some(loc), v.x, v.y, v.z) }
    }

    pub fn set_ivec3(&self, loc: &glow::NativeUniformLocation, x: i32, y: i32, z: i32) {
        unsafe { self.gl.uniform_3_i32(Some(loc), x, y, z) }
    }
}

impl Drop for ShaderProgram {
    fn drop(&mut self) {
        unsafe {
            self.gl.delete_program(self.id);
        }
    }
}
