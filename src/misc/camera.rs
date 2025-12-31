use glam::{Mat4, Vec3};

pub struct Camera {
    pub position: Vec3,
    pub front: Vec3,
    pub up: Vec3,
    pub right: Vec3,
    pub world_up: Vec3,

    pub projection: Mat4,
    pub yaw: f32,
    pub pitch: f32,
    pub mouse_sensitivity: f32,
    pub fov_deg: f32,
    pub near_d: f32,
    pub far_d: f32,
    pub ratio: f32,
}

impl Camera {
    pub fn new(position: Vec3, width: u32, height: u32) -> Self {
        let mut cam = Self {
            position,
            front: Vec3::new(0.0, 0.0, -1.0),
            up: Vec3::Y,
            right: Vec3::X,
            world_up: Vec3::Y,
            projection: Mat4::IDENTITY,
            yaw: 0.0,
            pitch: 0.0,
            mouse_sensitivity: 0.075,
            fov_deg: 80.0,
            near_d: 1.0,
            far_d: 10000.0,
            ratio: 1.0,
        };
        cam.handle_resolution(width, height);
        cam.update_camera_vectors();
        cam
    }

    pub fn handle_resolution(&mut self, width: u32, height: u32) {
        self.ratio = width as f32 / height as f32;
        self.projection = Mat4::perspective_rh_gl(self.fov_deg.to_radians(), self.ratio, self.near_d, self.far_d);
    }

    pub fn get_view_matrix(&self) -> Mat4 {
        let intra = self.position - self.position.floor();
        Mat4::look_at_rh(intra, intra + self.front, self.up)
    }

    pub fn process_mouse_movement(&mut self, x_offset: f32, y_offset: f32) {
        self.yaw += x_offset * self.mouse_sensitivity;
        self.pitch += y_offset * self.mouse_sensitivity;

        if self.pitch > 89.9 {
            self.pitch = 89.9;
        }
        if self.pitch < -89.9 {
            self.pitch = -89.9;
        }
        self.update_camera_vectors();
    }

    fn update_camera_vectors(&mut self) {
        let yaw = self.yaw.to_radians();
        let pitch = self.pitch.to_radians();

        let front = Vec3::new(yaw.cos() * pitch.cos(), pitch.sin(), yaw.sin() * pitch.cos());
        self.front = front.normalize();
        self.right = self.front.cross(self.world_up).normalize();
        self.up = self.right.cross(self.front).normalize();
    }
}
