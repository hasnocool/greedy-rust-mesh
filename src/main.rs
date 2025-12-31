use anyhow::{Context, Result};
use binary_greedy_mesher_demo_rs as demo;
use demo::data::{level_file::LevelFile, rle};
use demo::mesher::{mesh, MeshData, QuadData};
use demo::misc::{camera::Camera, shader::ShaderProgram};
use demo::rendering::chunk_renderer::{ChunkRenderer, DrawElementsIndirectCommand};
use demo::{parse_xyz_key, CS, CS_P3};
use glam::{IVec3, Vec3};
use glutin::config::ConfigTemplateBuilder;
use glutin::context::{ContextApi, ContextAttributesBuilder, Version};
use glutin::display::GetGlDisplay;
use glutin::prelude::*;
use glutin_winit::DisplayBuilder;
use glutin_winit::GlWindow;
use glow::HasContext;
use raw_window_handle::HasWindowHandle;
use rayon::prelude::*;
use std::env;
use std::rc::Rc;
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::time::Instant;
use winit::dpi::PhysicalSize;
use winit::event::{DeviceEvent, ElementState, Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, WindowAttributes};

const WINDOW_WIDTH: u32 = 1920;
const WINDOW_HEIGHT: u32 = 1080;
const DEFAULT_LEVEL_REL: &str = "levels/demo_terrain_96";

fn resolve_level_path() -> Result<PathBuf> {
    // Usage:
    //   cargo run -- [--level <path>]
    //   cargo run -- -l <path>
    // Paths that are not absolute are resolved relative to the crate root.
    let mut args = env::args().skip(1);
    let mut level: Option<PathBuf> = None;

    while let Some(a) = args.next() {
        match a.as_str() {
            "--level" | "-l" => {
                level = Some(PathBuf::from(
                    args.next().ok_or_else(|| anyhow::anyhow!("{a} requires a path"))?,
                ));
            }
            "--help" | "-h" => {
                eprintln!(
                    "Usage: binary_greedy_mesher_demo_rs [--level <path>]\n\nDefault: {DEFAULT_LEVEL_REL}\n"
                );
                std::process::exit(0);
            }
            _ => {
                // Ignore unknown args for now to avoid breaking existing workflows.
            }
        }
    }

    let level = level.unwrap_or_else(|| PathBuf::from(DEFAULT_LEVEL_REL));
    if level.is_absolute() {
        Ok(level)
    } else {
        Ok(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(level))
    }
}

const VERT_SRC: &str = r#"#version 460 core

struct QuadData {
  uint quadData1;
  uint quadData2;
};

layout(binding = 0, std430) readonly buffer ssbo1 {
  QuadData data[];
};

uniform mat4 u_view;
uniform mat4 u_projection;

uniform ivec3 eye_position_int;

out VS_OUT {
  out vec3 pos;
  flat vec3 normal;
  flat vec3 color;
} vs_out;

const vec3 normalLookup[6] = {
  vec3( 0, 1, 0 ),
  vec3(0, -1, 0 ),
  vec3( 1, 0, 0 ),
  vec3( -1, 0, 0 ),
  vec3( 0, 0, 1 ),
  vec3( 0, 0, -1 )
};

const vec3 colorLookup[8] = {
  vec3(0.2, 0.659, 0.839),
  vec3(0.302, 0.302, 0.302),
  vec3(0.278, 0.600, 0.141),
  vec3(0.1, 0.1, 0.6),
  vec3(0.1, 0.6, 0.6),
  vec3(0.6, 0.1, 0.6),
  vec3(0.6, 0.6, 0.1),
  vec3(0.6, 0.1, 0.1)
};

const int flipLookup[6] = int[6](1, -1, -1, 1, -1, 1);

void main() {
  ivec3 chunkOffsetPos = ivec3(gl_BaseInstance&255u, gl_BaseInstance>>8&255u, gl_BaseInstance>>16&255u) * 62;
  uint face = gl_BaseInstance>>24;

  int vertexID = int(gl_VertexID&3u);
  uint ssboIndex = gl_VertexID >> 2u;

  uint quadData1 = data[ssboIndex].quadData1;
  uint quadData2 = data[ssboIndex].quadData2;

  ivec3 iVertexPos = ivec3(quadData1, quadData1 >> 6u, quadData1 >> 12u) & 63;
  iVertexPos += chunkOffsetPos;

  int w = int((quadData1 >> 18u)&63u), h = int((quadData1 >> 24u)&63u);
  uint wDir = (face & 2) >> 1, hDir = 2 - (face >> 2);
  int wMod = vertexID >> 1, hMod = vertexID & 1;

  iVertexPos[wDir] += (w * wMod * flipLookup[face]);
  iVertexPos[hDir] += (h * hMod);

  vs_out.pos = iVertexPos;
  vs_out.normal = normalLookup[face];
  vs_out.color = colorLookup[(quadData2&255u) - 1];

  vec3 vertexPos = iVertexPos - eye_position_int;
  vertexPos[wDir] += 0.0007 * flipLookup[face] * (wMod * 2 - 1);
  vertexPos[hDir] += 0.0007 * (hMod * 2 - 1);

  gl_Position = u_projection * u_view * vec4(vertexPos, 1);
}
"#;

const FRAG_SRC: &str = r#"#version 460 core

layout(location=0) out vec3 out_color;

in VS_OUT {
  vec3 pos;
  flat vec3 normal;
  flat vec3 color;
} fs_in;

uniform vec3 eye_position;

const vec3 diffuse_color = vec3(0.15, 0.15, 0.15);
const vec3 rim_color = vec3(0.04, 0.04, 0.04);
const vec3 sun_position = vec3(250.0, 1000.0, 750.0) * 10000;

void main() {
  vec3 L = normalize(sun_position - fs_in.pos);
  vec3 V = normalize(eye_position - fs_in.pos);

  float rim = 1 - max(dot(V, fs_in.normal), 0.0);
  rim = smoothstep(0.6, 1.0, rim);

  out_color =
    fs_in.color +
    (diffuse_color * max(0, dot(L, fs_in.normal))) +
    (rim_color * vec3(rim, rim, rim))
  ;
}
"#;

#[derive(Clone)]
struct ChunkMesh {
    chunk_pos: IVec3,
    faces: [Vec<QuadData>; 6],
}

fn main() -> Result<()> {
    // --- Window + GL context (winit + glutin) ---
    let event_loop = EventLoop::new()?;
    let window_attributes = WindowAttributes::default()
        .with_title("Binary Greedy Meshing (Rust)")
        .with_inner_size(PhysicalSize::new(WINDOW_WIDTH, WINDOW_HEIGHT));

    let template = ConfigTemplateBuilder::new().with_alpha_size(8).with_depth_size(24);
    let display_builder = DisplayBuilder::new().with_window_attributes(Some(window_attributes));

    let (window, gl_config) = display_builder
        .build(&event_loop, template, |configs| {
            configs
                .reduce(|accum, config| {
                    if config.num_samples() > accum.num_samples() {
                        config
                    } else {
                        accum
                    }
                })
                .unwrap()
        })
        .map_err(|e| anyhow::anyhow!("Failed to create window / choose GL config: {e}"))?;

    let window = window.context("Window was not created")?;
    window.set_cursor_visible(false);
    let _ = window.set_cursor_grab(CursorGrabMode::Confined).or_else(|_| window.set_cursor_grab(CursorGrabMode::Locked));

    let raw_window_handle = window
        .window_handle()
        .ok()
        .map(|h| h.as_raw());
    let gl_display = gl_config.display();

    let context_attributes = ContextAttributesBuilder::new()
        .with_context_api(ContextApi::OpenGl(Some(Version::new(4, 4))))
        .build(raw_window_handle);

    let not_current_gl_context = unsafe {
        gl_display
            .create_context(&gl_config, &context_attributes)
            .context("create_context")?
    };

    let attrs = window
        .build_surface_attributes(<_>::default())
        .map_err(|e| anyhow::anyhow!("build_surface_attributes failed: {e}"))?;
    let gl_surface = unsafe {
        gl_display
            .create_window_surface(&gl_config, &attrs)
            .context("create_window_surface")?
    };

    let gl_context = not_current_gl_context
        .make_current(&gl_surface)
        .context("make_current")?;

    let gl = Rc::new(unsafe {
        glow::Context::from_loader_function(|s| {
            gl_display.get_proc_address(std::ffi::CString::new(s).unwrap().as_c_str()) as *const _
        })
    });

    unsafe {
        gl.enable(glow::DEPTH_TEST);
        gl.front_face(glow::CCW);
        gl.cull_face(glow::BACK);
        gl.enable(glow::CULL_FACE);
        gl.clear_color(0.529, 0.808, 0.922, 0.0);
        gl.enable(glow::MULTISAMPLE);
        gl.viewport(0, 0, WINDOW_WIDTH as i32, WINDOW_HEIGHT as i32);
    }

    // --- Shader + renderer ---
    let shader = ShaderProgram::new(&gl, VERT_SRC, FRAG_SRC).context("compile shaders")?;
    let u_proj = shader.uniform_location("u_projection").context("missing u_projection")?;
    let u_view = shader.uniform_location("u_view").context("missing u_view")?;
    let u_eye = shader.uniform_location("eye_position").context("missing eye_position")?;
    let u_eye_int = shader.uniform_location("eye_position_int").context("missing eye_position_int")?;

    let mut renderer = ChunkRenderer::new(&gl).context("create renderer")?;

    // --- Load level file ---
    let level_path = resolve_level_path()?;
    let mut level = LevelFile::default();
    level.load_from_file(&level_path)?;

    // Camera matches the C++ initial placement (roughly)
    let cam_start = Vec3::new(
        (level.size() as f32 * CS as f32) / 2.0,
        100.0,
        (level.size() as f32 * CS as f32) / 2.0 - 30.0,
    );
    let mut camera = Camera::new(cam_start, WINDOW_WIDTH, WINDOW_HEIGHT);

    // --- Mesh all chunks (parallel compute, sequential upload) ---
    let chunk_meshes: Vec<ChunkMesh> = level
        .chunk_table
        .par_iter()
        .map(|entry| {
            let (x, y, z) = parse_xyz_key(entry.key);
            let chunk_pos = IVec3::new(x as i32, y as i32, z as i32);

            let mut voxels = vec![0u8; CS_P3];
            let mut mesh_data = MeshData::new(10_000);
            mesh_data.opaque_mask.fill(0);

            let start = entry.rle_data_begin as usize;
            let end = start + entry.rle_data_size as usize;
            let rle_slice = &level.buffer[start..end];
            rle::decompress_to_voxels_and_opaque_mask(rle_slice, &mut voxels, &mut mesh_data.opaque_mask);
            mesh(&voxels, &mut mesh_data);

            let faces: [Vec<QuadData>; 6] = std::array::from_fn(|face| {
                let begin = mesh_data.face_vertex_begin[face];
                let len = mesh_data.face_vertex_length[face];
                if len == 0 {
                    Vec::new()
                } else {
                    mesh_data.vertices[begin..begin + len].to_vec()
                }
            });

            ChunkMesh { chunk_pos, faces }
        })
        .collect();

    // Upload and keep indirect commands per chunk/face.
    let mut per_chunk_cmds: Vec<(IVec3, [Option<DrawElementsIndirectCommand>; 6])> = Vec::with_capacity(chunk_meshes.len());
    for cm in chunk_meshes {
        let mut cmds: [Option<DrawElementsIndirectCommand>; 6] = std::array::from_fn(|_| None);
        for face in 0..6u32 {
            let quads = &cm.faces[face as usize];
            if quads.is_empty() {
                continue;
            }
            let base_vertex = renderer.upload_quads(quads)?;
            let base_instance = ((face as u32) << 24)
                | ((cm.chunk_pos.z as u32) << 16)
                | ((cm.chunk_pos.y as u32) << 8)
                | (cm.chunk_pos.x as u32);

            cmds[face as usize] = Some(DrawElementsIndirectCommand {
                index_count: (quads.len() as u32) * 6,
                instance_count: 1,
                first_index: 0,
                base_vertex,
                base_instance,
            });
        }
        per_chunk_cmds.push((cm.chunk_pos, cmds));
    }

    // --- Main loop ---
    let mut last_frame = Instant::now();
    let noclip_speed: f32 = 250.0;
    let noclip_fast_multiplier: f32 = 4.0;
    let mut wireframe = false;

    #[derive(Default, Copy, Clone)]
    struct InputState {
        w: bool,
        a: bool,
        s: bool,
        d: bool,
        shift: bool,
    }

    let mut input = InputState::default();

    // Mouse deltas from raw device events
    let mut mouse_dx: f32 = 0.0;
    let mut mouse_dy: f32 = 0.0;

    let _ = event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Poll);

        match event {
            Event::DeviceEvent {
                event: DeviceEvent::MouseMotion { delta },
                ..
            } => {
                mouse_dx += delta.0 as f32;
                mouse_dy += delta.1 as f32;
            }

            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => elwt.exit(),
                WindowEvent::Resized(size) => {
                    let _ = gl_surface.resize(
                        &gl_context,
                        NonZeroU32::new(size.width.max(1)).unwrap(),
                        NonZeroU32::new(size.height.max(1)).unwrap(),
                    );
                    camera.handle_resolution(size.width.max(1), size.height.max(1));
                    unsafe { gl.viewport(0, 0, size.width as i32, size.height as i32) };
                }
                WindowEvent::KeyboardInput { event: key_event, .. } => {
                    if let PhysicalKey::Code(code) = key_event.physical_key {
                        let pressed = key_event.state == ElementState::Pressed;

                        match code {
                            KeyCode::KeyW => input.w = pressed,
                            KeyCode::KeyA => input.a = pressed,
                            KeyCode::KeyS => input.s = pressed,
                            KeyCode::KeyD => input.d = pressed,
                            KeyCode::ShiftLeft | KeyCode::ShiftRight => input.shift = pressed,
                            _ => {}
                        }

                        if code == KeyCode::Escape && key_event.state == ElementState::Released {
                            elwt.exit();
                        }
                        if code == KeyCode::KeyX && key_event.state == ElementState::Released {
                            wireframe = !wireframe;
                            unsafe {
                                gl.polygon_mode(
                                    glow::FRONT_AND_BACK,
                                    if wireframe { glow::LINE } else { glow::FILL },
                                );
                            }
                        }
                    }
                }
                _ => {}
            },

            Event::AboutToWait => {
                // Timing
                let now = Instant::now();
                let dt = (now - last_frame).as_secs_f32();
                last_frame = now;

                // Apply mouse look
                if mouse_dx != 0.0 || mouse_dy != 0.0 {
                    camera.process_mouse_movement(mouse_dx, -mouse_dy);
                    mouse_dx = 0.0;
                    mouse_dy = 0.0;
                }

                // Fly camera (WASD + Shift)
                let forward = (input.w as i32 - input.s as i32) as f32;
                let right = (input.d as i32 - input.a as i32) as f32;

                let mut wishdir = (camera.front * forward) + (camera.right * right);
                if wishdir.length_squared() > 0.0 {
                    wishdir = wishdir.normalize();
                    let speed = noclip_speed * if input.shift { noclip_fast_multiplier } else { 1.0 };
                    camera.position += wishdir * speed * dt;
                }

                // Render
                unsafe { gl.clear(glow::COLOR_BUFFER_BIT | glow::DEPTH_BUFFER_BIT) };

                shader.bind();
                shader.set_mat4(&u_proj, &camera.projection);
                shader.set_mat4(&u_view, &camera.get_view_matrix());
                shader.set_vec3(&u_eye, &camera.position);

                let eye_int = camera.position.floor();
                shader.set_ivec3(&u_eye_int, eye_int.x as i32, eye_int.y as i32, eye_int.z as i32);

                let camera_chunk_pos = (camera.position / (CS as f32)).floor();
                let camera_chunk_pos = IVec3::new(camera_chunk_pos.x as i32, camera_chunk_pos.y as i32, camera_chunk_pos.z as i32);

                for (chunk_pos, cmds) in &per_chunk_cmds {
                    for face in 0..6 {
                        if let Some(cmd) = cmds[face] {
                            let visible = match face {
                                0 => camera_chunk_pos.y >= chunk_pos.y,
                                1 => camera_chunk_pos.y <= chunk_pos.y,
                                2 => camera_chunk_pos.x >= chunk_pos.x,
                                3 => camera_chunk_pos.x <= chunk_pos.x,
                                4 => camera_chunk_pos.z >= chunk_pos.z,
                                5 => camera_chunk_pos.z <= chunk_pos.z,
                                _ => true,
                            };
                            if visible {
                                renderer.add_draw_command(cmd);
                            }
                        }
                    }
                }

                renderer.render();

                gl_surface.swap_buffers(&gl_context).expect("swap_buffers");

                let _ = dt;
            }
            _ => {}
        }
    });

    #[allow(unreachable_code)]
    Ok(())
}
