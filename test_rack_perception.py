"""Fast image-only geometry/tracking tests; no Isaac or object-state API exists."""
import unittest
from types import SimpleNamespace
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from isaac_rack_markers import RACK_MARKERS, TUBE_UPPER_OFFSET, TUBE_LOWER_OFFSET
from isaac_rack_perception import MarkerPerception


def rendered_camera(eye, spheres):
    eye = np.array(eye)
    forward = np.array([.49, -.02, .1]) - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 0., 1.])
    right /= np.linalg.norm(right)
    matrix = np.column_stack((right, np.cross(forward, right), forward))
    quaternion_xyzw = Rotation.from_matrix(matrix).as_quat()
    k = np.array([[1000., 0., 320.], [0., 1000., 240.], [0., 0., 1.]], dtype=np.float32)
    v, u = np.mgrid[:480, :640]
    rays = np.stack(((u + .5 - 320.) / 1000., (v + .5 - 240.) / 1000., np.ones_like(u)), -1)
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    depth = np.full((480, 640), 3., dtype=np.float32)
    a = (rays ** 2).sum(-1)
    for position, radius, color in spheres:
        center = (position - eye) @ matrix
        b = -2. * (rays * center).sum(-1)
        c = center @ center - radius ** 2
        discriminant = b * b - 4. * a * c
        z = (-b - np.sqrt(discriminant.clip(0.))) / (2. * a)
        mask = (discriminant > 0.) & (z > .05) & (z < depth)
        depth[mask] = z[mask]
        rgb[mask] = color
    return SimpleNamespace(data=SimpleNamespace(output={"rgb": torch.tensor(rgb[None]),
        "distance_to_image_plane": torch.tensor(depth[None, ..., None])}, intrinsic_matrices=torch.tensor(k[None]),
        quat_w_ros=torch.tensor(np.r_[quaternion_xyzw[3], quaternion_xyzw[:3]][None]),
        pos_w=torch.tensor(eye[None])))


class PerceptionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.env = SimpleNamespace(scene={}, device="cpu")
        self.env.unwrapped = self.env
        self.perception = MarkerPerception(self.env)

    def images(self, offset=0., hidden=False, rack_hidden=False):
        tube = np.array([.43 + offset, -.16, .08])
        rack = np.array([.55 + offset, .12 - offset, 0.])
        rotation = Rotation.from_euler('z', .12).as_matrix()
        spheres = [(rack + rotation @ p, .007, (255, 40, 0)) for p in np.array(RACK_MARKERS)]
        if rack_hidden:
            spheres = []
        if not hidden:
            spheres += [(tube + [0., 0., TUBE_UPPER_OFFSET], .012, (255, 0, 255)),
                        (tube + [0., 0., TUBE_LOWER_OFFSET], .012, (0, 255, 0))]
        for name, eye in (("rack_camera_a", (.75, -.55, .65)), ("rack_camera_b", (.15, -.40, .72))):
            self.env.scene[name] = rendered_camera(eye, spheres)
        goal = rack + rotation @ np.array([.0136571, -.0136573, .0777])
        return tube, goal

    def test_pose_target_and_image_tracking_follow_motion(self):
        for offset in (0., .003, .007, .012):
            tube, goal = self.images(offset)
            measured, target = self.perception.estimate()
            np.testing.assert_allclose(measured[0, :3], tube, atol=.0004)
            np.testing.assert_allclose(target[0], goal, atol=.0004)
        self.assertTrue(self.perception.regions)

    def test_missing_markers_are_rejected_not_replaced_by_truth(self):
        self.images()
        self.perception.estimate()
        self.images(hidden=True)
        with self.assertRaisesRegex(ValueError, "occluded"):
            self.perception.estimate()

    def test_static_fixture_calibration_survives_occlusion(self):
        self.images()
        _, original = self.perception.estimate()
        tube, _ = self.images(offset=.007, rack_hidden=True)
        pose, target = self.perception.estimate()
        np.testing.assert_allclose(pose[0, :3], tube, atol=.0004)
        np.testing.assert_array_equal(target, original)
        self.perception.rack_destination = None
        with self.assertRaisesRegex(ValueError, "three visible"):
            self.perception.estimate()


if __name__ == "__main__":
    unittest.main()
