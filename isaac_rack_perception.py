"""RGB-D marker geometry only: never reads object, rack or marker USD poses."""
import itertools
import numpy as np
import torch
from scipy.ndimage import label, find_objects
from scipy.ndimage import binary_erosion
from isaac_rack_markers import RACK_MARKERS, TUBE_UPPER_OFFSET, TUBE_LOWER_OFFSET


def camera_snapshot(camera, regions=None, calibration=None):
    data = camera.data
    rgb = data.output["rgb"][0, ..., :3].cpu().numpy()
    depth = data.output["distance_to_image_plane"][0].cpu().numpy().squeeze()
    masks = {}
    for color in ("upper", "lower", "rack"):
        v0, v1, u0, u1 = (regions or {}).get(color, (0, rgb.shape[0], 0, rgb.shape[1]))
        r, g, b = rgb[v0:v1, u0:u1].astype(np.uint16).transpose(2, 0, 1)
        if color == "upper":
            mask = (r > 77) & (b > 77) & (g * 100 < 65 * np.minimum(r, b))
        elif color == "lower":
            mask = (g > 77) & (g * 10 > 16 * r) & (g * 10 > 16 * b)
        else:
            mask = (r > 90) & (r * 100 > 145 * g) & (g * 10 > 17 * b)
        masks[color] = mask, v0, u0
    if calibration is None:
        k = data.intrinsic_matrices[0].cpu().numpy()
        w, x, y, z = data.quat_w_ros[0].cpu().numpy()
        rotation = np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                             [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                             [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])
        origin = data.pos_w[0].cpu().numpy()
    else:
        k, rotation, origin = calibration
    return masks, depth, k, rotation, origin


def sphere_centers(camera, color, radius, debug=False, snapshot=None):
    masks, depth, k, rotation, origin = camera_snapshot(camera) if snapshot is None else snapshot
    mask, v0, u0 = masks[color]
    crop_depth = depth[v0:v0+mask.shape[0], u0:u0+mask.shape[1]]
    labels, count = label(mask & np.isfinite(crop_depth) & (crop_depth > .05) & (crop_depth < 2.))
    candidates = []
    for component, bounds in enumerate(find_objects(labels), 1):
        if bounds is None:
            continue
        region = labels[bounds] == component
        if region.sum() < 30:
            continue
        v, u = np.where(binary_erosion(region, iterations=1))
        v += bounds[0].start + v0
        u += bounds[1].start + u0
        if len(u) < 25:
            continue
        z = depth[v, u]
        points = np.column_stack(((u + .5 - k[0, 2]) * z / k[0, 0], (v + .5 - k[1, 2]) * z / k[1, 1], z))
        # Fit the center of a known-radius sphere to visible surface points.
        mean = np.median(points, axis=0)
        points = points[np.linalg.norm(points - mean, axis=1) < radius * 2.5]
        if len(points) < 20:
            continue
        center = mean + mean / np.linalg.norm(mean) * radius * .5
        # Bounded robust Gauss-Newton with an analytic 3-D Jacobian avoids
        # expensive generic numerical optimization on every video frame.
        for _ in range(10):
            delta = center - points
            distances = np.linalg.norm(delta, axis=1).clip(1.e-8)
            errors = distances - radius
            jacobian = delta / distances[:, None]
            weights = 1. / np.sqrt(1. + (errors / .0003) ** 2)
            change = np.linalg.solve(jacobian.T @ (jacobian * weights[:, None]) + np.eye(3) * 1.e-6,
                                     jacobian.T @ (weights * errors))
            center -= change * min(1., .004 / max(np.linalg.norm(change), 1.e-8))
            if np.linalg.norm(change) < 1.e-7:
                break
        fitted_radius = radius
        residual = np.abs(np.linalg.norm(points - center, axis=1) - radius).mean()
        if debug:
            print("SPHERE_FIT", color, len(u), fitted_radius, residual, flush=True)
        if abs(fitted_radius - radius) > .0015 or residual > .0007:
            continue
        candidates.append((len(u), center @ rotation.T + origin))
    return [center for _, center in sorted(candidates, key=lambda item: -item[0])]


class MarkerPerception:
    def __init__(self, env):
        self.env = env.unwrapped
        self.last_error = "Waiting for camera frames"
        self.regions = {}
        self.calibrations = {}
        self.rack_destination = None

    def estimate(self):
        detections = {key: [] for key in ("upper", "lower", "rack")}
        for name in ("rack_camera_a", "rack_camera_b"):
            camera = self.env.scene[name]
            snapshot = camera_snapshot(camera, self.regions.get(name), self.calibrations.get(name))
            self.calibrations[name] = snapshot[2:]
            next_regions = {}
            for color in detections:
                radius = .007 if color == "rack" else .012
                centers = sphere_centers(camera, color, radius, snapshot=snapshot)
                detections[color] += centers
                if centers:
                    _, depth, k, rotation, origin = snapshot
                    local = (np.array(centers) - origin) @ rotation
                    pixels = local[:, :2] / local[:, 2:] * np.array([k[0, 0], k[1, 1]]) + k[:2, 2] - .5
                    padding = int(k[0, 0] * radius / local[:, 2].min()) + 30
                    low = np.floor(pixels.min(0) - padding).astype(int)
                    high = np.ceil(pixels.max(0) + padding).astype(int)
                    next_regions[color] = (max(0, low[1]), min(depth.shape[0], high[1]),
                                           max(0, low[0]), min(depth.shape[1], high[0]))
            # Track only image-derived regions. A lost component triggers a
            # full-frame search next time, never a truth-based search window.
            self.regions[name] = next_regions
        if not detections["upper"] or not detections["lower"]:
            raise ValueError("Tube markers are occluded or uncertain")
        for color in ("upper", "lower"):
            if len(detections[color]) > 2:
                raise ValueError("Ambiguous tube marker detections")
            if len(detections[color]) == 2 and np.linalg.norm(detections[color][0] - detections[color][1]) > .003:
                raise ValueError("Camera estimates disagree")
        upper = np.mean(detections["upper"], axis=0)
        lower = np.mean(detections["lower"], axis=0)
        length = np.linalg.norm(upper - lower)
        if abs(length - (TUBE_UPPER_OFFSET - TUBE_LOWER_OFFSET)) > .0015:
            raise ValueError("Tube marker spacing inconsistent")
        up = (upper - lower) / length
        if up[2] < .75:
            raise ValueError("Tube tilt exceeds the supported upright grasp")
        axis = np.cross((0., 0., 1.), up)
        quaternion = np.r_[1. + up[2], axis]
        quaternion /= np.linalg.norm(quaternion)
        # Merge repeated detections from the two cameras without using truth.
        clusters = []
        for point in detections["rack"]:
            match = next((cluster for cluster in clusters if np.linalg.norm(np.mean(cluster, axis=0) - point) < .004), None)
            if match is None:
                clusters.append([point])
            else:
                match.append(point)
        centers = np.array([np.mean(cluster, axis=0) for cluster in clusters])
        if len(centers) < 3:
            if self.rack_destination is None:
                raise ValueError("Rack needs three visible calibrated markers")
            # The fixture is static during an episode. Retain only a position
            # previously measured from images, never a simulator pose.
            center = (upper + lower) / 2. - (TUBE_UPPER_OFFSET + TUBE_LOWER_OFFSET) / 2. * up
            self.last_error = "Tube detected; static rack uses prior visual calibration"
            return (torch.tensor(np.r_[center, quaternion][None], device=self.env.device, dtype=torch.float32),
                    torch.tensor(self.rack_destination[None], device=self.env.device, dtype=torch.float32))
        if len(centers) > 6:
            raise ValueError("Ambiguous rack marker detections")
        template = np.array(RACK_MARKERS)
        best = None
        for indices in itertools.permutations(range(len(centers)), 3):
            observed = centers[list(indices)]
            a, b = template - template.mean(0), observed - observed.mean(0)
            u, _, vt = np.linalg.svd(a.T @ b)
            rotation = vt.T @ np.diag([1., 1., np.linalg.det(vt.T @ u.T)]) @ u.T
            translation = observed.mean(0) - template.mean(0) @ rotation.T
            error = np.linalg.norm(template @ rotation.T + translation - observed, axis=1).max()
            if best is None or error < best[0]:
                best = error, rotation, translation
        error, rotation, translation = best
        if error > .001 or rotation[2, 2] < .9:
            raise ValueError("Rack marker geometry inconsistent")
        destination = np.array([.0136571, -.0136573, .0777]) @ rotation.T + translation
        self.rack_destination = destination.copy()
        center = (upper + lower) / 2. - (TUBE_UPPER_OFFSET + TUBE_LOWER_OFFSET) / 2. * up
        pose = torch.tensor(np.r_[center, quaternion][None], device=self.env.device, dtype=torch.float32)
        target = torch.tensor(destination[None], device=self.env.device, dtype=torch.float32)
        self.last_error = "RGB-D markers detected"
        return pose, target
