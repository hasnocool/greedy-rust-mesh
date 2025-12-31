use crate::{CS, CS_2, CS_P, CS_P2};
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
pub struct QuadData {
    pub quad_data1: u32,
    pub quad_data2: u32,
}

#[derive(Debug)]
pub struct MeshData {
    pub face_masks: Vec<u64>,      // CS_2 * 6
    pub opaque_mask: Vec<u64>,     // CS_P2
    pub forward_merged: Vec<u8>,   // faces 0-3: CS; faces 4-5: CS_2
    pub right_merged: Vec<u8>,     // faces 4-5: CS
    pub vertices: Vec<QuadData>,
    pub face_vertex_begin: [usize; 6],
    pub face_vertex_length: [usize; 6],
}

impl MeshData {
    pub fn new(initial_quads: usize) -> Self {
        Self {
            face_masks: vec![0u64; CS_2 * 6],
            opaque_mask: vec![0u64; CS_P2],
            forward_merged: vec![0u8; CS_2.max(CS)],
            right_merged: vec![0u8; CS],
            vertices: vec![QuadData::default(); initial_quads],
            face_vertex_begin: [0; 6],
            face_vertex_length: [0; 6],
        }
    }

    pub fn clear_runtime(&mut self) {
        self.vertices.fill(QuadData::default());
        self.face_masks.fill(0);
        self.forward_merged.fill(0);
        self.right_merged.fill(0);
        self.face_vertex_begin = [0; 6];
        self.face_vertex_length = [0; 6];
    }
}

#[inline]
fn get_axis_index(axis: usize, a: usize, b: usize, c: usize) -> usize {
    if axis == 0 {
        b + (a * CS_P) + (c * CS_P2)
    } else if axis == 1 {
        b + (c * CS_P) + (a * CS_P2)
    } else {
        c + (a * CS_P) + (b * CS_P2)
    }
}

#[inline]
fn get_quad(x: u32, y: u32, z: u32, w: u32, h: u32, ty: u32) -> QuadData {
    let quad_data1 = (h << 24) | (w << 18) | (z << 12) | (y << 6) | x;
    QuadData {
        quad_data1,
        quad_data2: ty,
    }
}

const P_MASK: u64 = !(1u64 << 63 | 1);

pub fn mesh(voxels: &[u8], mesh: &mut MeshData) {
    // Reset counts
    let mut vertex_i: usize = 0;

    // Hidden face culling
    for a in 1..(CS_P - 1) {
        let a_cs_p = a * CS_P;
        for b in 1..(CS_P - 1) {
            let column_bits = mesh.opaque_mask[(a * CS_P) + b] & P_MASK;
            let ba_index = (b - 1) + (a - 1) * CS;
            let ab_index = (a - 1) + (b - 1) * CS;

            mesh.face_masks[ba_index + 0 * CS_2] = (column_bits & !mesh.opaque_mask[a_cs_p + CS_P + b]) >> 1;
            mesh.face_masks[ba_index + 1 * CS_2] = (column_bits & !mesh.opaque_mask[a_cs_p - CS_P + b]) >> 1;

            mesh.face_masks[ab_index + 2 * CS_2] = (column_bits & !mesh.opaque_mask[a_cs_p + (b + 1)]) >> 1;
            mesh.face_masks[ab_index + 3 * CS_2] = (column_bits & !mesh.opaque_mask[a_cs_p + (b - 1)]) >> 1;

            mesh.face_masks[ba_index + 4 * CS_2] = column_bits & !(mesh.opaque_mask[a_cs_p + b] >> 1);
            mesh.face_masks[ba_index + 5 * CS_2] = column_bits & !(mesh.opaque_mask[a_cs_p + b] << 1);
        }
    }

    // Faces 0-3
    for face in 0..4usize {
        let axis = face / 2;
        let face_vertex_begin = vertex_i;

        for layer in 0..CS {
            let bits_location = layer * CS + face * CS_2;

            for forward in 0..CS {
                let mut bits_here = mesh.face_masks[forward + bits_location];
                if bits_here == 0 {
                    continue;
                }

                let bits_next = if forward + 1 < CS {
                    mesh.face_masks[(forward + 1) + bits_location]
                } else {
                    0
                };

                let mut right_merged_run: u8 = 1;
                while bits_here != 0 {
                    let bit_pos = bits_here.trailing_zeros() as usize;

                    let ty = voxels[get_axis_index(axis, forward + 1, bit_pos + 1, layer + 1)] as u32;
                    let mut forward_merged_val = mesh.forward_merged[bit_pos];

                    if ((bits_next >> bit_pos) & 1) == 1
                        && ty == voxels[get_axis_index(axis, forward + 2, bit_pos + 1, layer + 1)] as u32
                    {
                        forward_merged_val = forward_merged_val.saturating_add(1);
                        mesh.forward_merged[bit_pos] = forward_merged_val;
                        bits_here &= !(1u64 << bit_pos);
                        continue;
                    }

                    for right in (bit_pos + 1)..CS {
                        if ((bits_here >> right) & 1) == 0 {
                            break;
                        }
                        if forward_merged_val != mesh.forward_merged[right] {
                            break;
                        }
                        if ty != voxels[get_axis_index(axis, forward + 1, right + 1, layer + 1)] as u32 {
                            break;
                        }
                        mesh.forward_merged[right] = 0;
                        right_merged_run = right_merged_run.saturating_add(1);
                    }

                    bits_here &= !((1u64 << (bit_pos + right_merged_run as usize)) - 1);

                    let mesh_front = (forward as i32) - (forward_merged_val as i32);
                    let mesh_left = bit_pos as i32;
                    let mesh_up = (layer as i32) + ((!face & 1) as i32);

                    let mesh_width = right_merged_run as u32;
                    let mesh_length = (forward_merged_val as u32) + 1;

                    mesh.forward_merged[bit_pos] = 0;
                    right_merged_run = 1;

                    let quad = match face {
                        0 | 1 => get_quad(
                            (mesh_front as u32) + if face == 1 { mesh_length } else { 0 },
                            mesh_up as u32,
                            mesh_left as u32,
                            mesh_length,
                            mesh_width,
                            ty,
                        ),
                        2 | 3 => get_quad(
                            mesh_up as u32,
                            (mesh_front as u32) + if face == 2 { mesh_length } else { 0 },
                            mesh_left as u32,
                            mesh_length,
                            mesh_width,
                            ty,
                        ),
                        _ => unreachable!(),
                    };

                    if vertex_i >= mesh.vertices.len() {
                        mesh.vertices.resize(mesh.vertices.len().max(1) * 2, QuadData::default());
                    }
                    mesh.vertices[vertex_i] = quad;
                    vertex_i += 1;
                }
            }
        }

        let face_vertex_length = vertex_i - face_vertex_begin;
        mesh.face_vertex_begin[face] = face_vertex_begin;
        mesh.face_vertex_length[face] = face_vertex_length;
    }

    // Faces 4-5
    for face in 4..6usize {
        let axis = face / 2;
        let face_vertex_begin = vertex_i;

        for forward in 0..CS {
            let bits_location = forward * CS + face * CS_2;
            let bits_forward_location = (forward + 1) * CS + face * CS_2;

            for right in 0..CS {
                let mut bits_here = mesh.face_masks[right + bits_location];
                if bits_here == 0 {
                    continue;
                }

                let bits_forward = if forward < CS - 1 {
                    mesh.face_masks[right + bits_forward_location]
                } else {
                    0
                };

                let bits_right = if right < CS - 1 {
                    mesh.face_masks[right + 1 + bits_location]
                } else {
                    0
                };

                let right_cs = right * CS;

                while bits_here != 0 {
                    let bit_pos = bits_here.trailing_zeros() as usize;
                    bits_here &= !(1u64 << bit_pos);

                    let ty = voxels[get_axis_index(axis, right + 1, forward + 1, bit_pos)] as u32;

                    let f_idx = right_cs + (bit_pos - 1);
                    let mut forward_merged_val = mesh.forward_merged[f_idx];
                    let mut right_merged_val = mesh.right_merged[bit_pos - 1];

                    if right_merged_val == 0
                        && ((bits_forward >> bit_pos) & 1) == 1
                        && ty == voxels[get_axis_index(axis, right + 1, forward + 2, bit_pos)] as u32
                    {
                        forward_merged_val = forward_merged_val.saturating_add(1);
                        mesh.forward_merged[f_idx] = forward_merged_val;
                        continue;
                    }

                    let next_forward_merged = if right + 1 < CS {
                        mesh.forward_merged[(right_cs + CS) + (bit_pos - 1)]
                    } else {
                        0
                    };

                    if ((bits_right >> bit_pos) & 1) == 1
                        && forward_merged_val == next_forward_merged
                        && ty == voxels[get_axis_index(axis, right + 2, forward + 1, bit_pos)] as u32
                    {
                        mesh.forward_merged[f_idx] = 0;
                        right_merged_val = right_merged_val.saturating_add(1);
                        mesh.right_merged[bit_pos - 1] = right_merged_val;
                        continue;
                    }

                    let mesh_left = (right as i32) - (right_merged_val as i32);
                    let mesh_front = (forward as i32) - (forward_merged_val as i32);
                    let mesh_up = (bit_pos as i32) - 1 + ((!face & 1) as i32);

                    let mesh_width = 1 + (right_merged_val as u32);
                    let mesh_length = 1 + (forward_merged_val as u32);

                    mesh.forward_merged[f_idx] = 0;
                    mesh.right_merged[bit_pos - 1] = 0;

                    // Match C++: for face 4 (positive direction), shift onto the far plane.
                    let x = (mesh_left as u32) + if face == 4 { mesh_width } else { 0 };
                    let quad = match face {
                        4 | 5 => get_quad(
                            x,
                            mesh_front as u32,
                            mesh_up as u32,
                            mesh_width,
                            mesh_length,
                            ty,
                        ),
                        _ => unreachable!(),
                    };

                    if vertex_i >= mesh.vertices.len() {
                        mesh.vertices.resize(mesh.vertices.len().max(1) * 2, QuadData::default());
                    }
                    mesh.vertices[vertex_i] = quad;
                    vertex_i += 1;
                }
            }
        }

        let face_vertex_length = vertex_i - face_vertex_begin;
        mesh.face_vertex_begin[face] = face_vertex_begin;
        mesh.face_vertex_length[face] = face_vertex_length;
    }

    // Shrink visible slice markers (we keep allocated capacity in vertices vec)
    // Caller uses face ranges to decide what to upload.
}
