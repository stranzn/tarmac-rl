import gymnasium as gym
import numpy as np
import pygame
import math
from stopwatch import Stopwatch
from track_builder import TrackBuilder


class CarEnv(gym.Env):
    """
    top-down car racing environment built on gymnasium.

    observation (12 floats, all in [-1, 1]):
        0   speed            / MAX_SPEED
        1   sin(heading)
        2   cos(heading)
        3   signed lateral distance to track centre / TRACK_WIDTH
        4   sin(heading_error)
        5   cos(heading_error)
        6-7 sin/cos of lookahead error  5 waypoints ahead
        8-9 sin/cos of lookahead error 15 waypoints ahead
       10-11 sin/cos of lookahead error 30 waypoints ahead

    actions (MultiDiscrete [3, 3]):
        axis 0 — steer:    0=left  1=straight  2=right
        axis 1 — throttle: 0=coast 1=accelerate 2=brake
    """

    metadata = {"render_modes": ["human"], "render_fps": 60}

    # ------------------------------------------------------------------ tuning
    MAX_SPEED       = 12.0
    STEER_RATE      = 0.045
    ACCEL           = 0.55
    BRAKE           = 0.50
    FRICTION        = 0.015
    TRACK_WIDTH     = 44
    OFF_ROAD_MARGIN = 10
    MAX_STEPS       = 4000
    # -----------------------------------------------------------------------

    def __init__(self, render_mode=None):
        super().__init__()

        self.SCREEN_WIDTH  = 1920
        self.SCREEN_HEIGHT = 1080

        self.render_mode = render_mode
        self.screen    = None
        self.clock     = None
        self.car_image = None

        # build track, resample to uniform spacing, then fit to screen
        builder  = TrackBuilder(density=1.0)
        self._tp = builder.build_monza_circuit()
        self._tp = self._resample_track(self._tp, target_spacing=15.0)
        self._normalize_track()

        self.action_space = gym.spaces.MultiDiscrete([3, 3])
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(12,), dtype=np.float32
        )

        self.stopwatch        = Stopwatch()
        self.current_lap_time = 0.0
        self.best_lap_time    = float('inf')

        self.reset()

    # ------------------------------------------------------------------ track

    # resamples waypoints to even spacing so lookahead offsets correspond
    # to consistent physical distances regardless of raw builder density
    def _resample_track(self, tp, target_spacing=15.0):
        dists = [0.0]
        for i in range(1, len(tp)):
            d = np.hypot(tp[i][0] - tp[i-1][0], tp[i][1] - tp[i-1][1])
            dists.append(dists[-1] + d)
        total_length = dists[-1]

        n_points     = int(total_length / target_spacing)
        sample_dists = np.linspace(0, total_length, n_points, endpoint=False)

        new_points = []
        j = 0
        for sd in sample_dists:
            while j < len(dists) - 1 and dists[j+1] < sd:
                j += 1
            seg_len = dists[j+1] - dists[j]
            t = 0.0 if seg_len < 1e-9 else (sd - dists[j]) / seg_len
            x = tp[j][0] + t * (tp[j+1][0] - tp[j][0])
            y = tp[j][1] + t * (tp[j+1][1] - tp[j][1])
            new_points.append([x, y])

        return np.array(new_points, dtype=np.float32)

    # scales and centres the track to fill the screen with a small padding margin
    def _normalize_track(self):
        min_x = np.min(self._tp[:, 0]);  max_x = np.max(self._tp[:, 0])
        min_y = np.min(self._tp[:, 1]);  max_y = np.max(self._tp[:, 1])

        padding  = 0.1
        usable_w = self.SCREEN_WIDTH  * (1 - 2 * padding)
        usable_h = self.SCREEN_HEIGHT * (1 - 2 * padding)
        scale    = min(usable_w / (max_x - min_x), usable_h / (max_y - min_y))

        self._tp *= scale

        offset_x = (self.SCREEN_WIDTH  - (np.max(self._tp[:,0]) + np.min(self._tp[:,0]))) / 2
        offset_y = (self.SCREEN_HEIGHT - (np.max(self._tp[:,1]) + np.min(self._tp[:,1]))) / 2
        self._tp[:, 0] += offset_x
        self._tp[:, 1] += offset_y
        self.TRACK_WIDTH *= scale

    # ------------------------------------------------------------------ reset

    # resets car to the start waypoint, matched to track direction so heading
    # error begins at zero rather than requiring the agent to self-correct immediately
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        start_idx = 10
        prev_idx  = (start_idx - 1) % len(self._tp)
        next_idx  = (start_idx + 1) % len(self._tp)
        tx = self._tp[next_idx][0] - self._tp[prev_idx][0]
        ty = self._tp[next_idx][1] - self._tp[prev_idx][1]
        spawn_angle = math.atan2(ty, tx)

        start      = self._tp[start_idx]
        self.x     = float(start[0])
        self.y     = float(start[1])
        self.angle = spawn_angle
        self.speed = 0.0
        self.done  = False
        self._step_count = 0

        self._last_track_idx = self._nearest_track_idx()

        self.stopwatch.start_lap(self._step_count)
        self.current_lap_time = 0.0

        return self._get_obs(), {}

    # ------------------------------------------------------------------ helpers

    # returns the index of the closest waypoint to the car's current position
    def _nearest_track_idx(self) -> int:
        dx = self._tp[:, 0] - self.x
        dy = self._tp[:, 1] - self.y
        return int(np.argmin(dx * dx + dy * dy))

    # returns signed lateral offset, track heading, and heading error relative
    # to the nearest waypoint; used by both obs and reward
    def _track_info(self):
        idx      = self._nearest_track_idx()
        prev_idx = (idx - 1) % len(self._tp)
        next_idx = (idx + 1) % len(self._tp)
        tx = float(self._tp[next_idx][0] - self._tp[prev_idx][0])
        ty = float(self._tp[next_idx][1] - self._tp[prev_idx][1])
        norm = math.hypot(tx, ty) + 1e-9
        tx /= norm;  ty /= norm

        ex = self.x - float(self._tp[idx][0])
        ey = self.y - float(self._tp[idx][1])

        signed_lat  = tx * ey - ty * ex
        track_angle = math.atan2(ty, tx)
        heading_err = (self.angle - track_angle + math.pi) % (2 * math.pi) - math.pi
        dist        = math.hypot(ex, ey)
        return dist, signed_lat, heading_err

    # computes heading errors toward waypoints 5, 15, and 30 steps ahead.
    # shared by _get_obs() and step() to keep values consistent.
    def _lookahead_errors(self, idx: int) -> tuple:
        n = len(self._tp)

        def err_to(offset):
            future_idx   = (idx + offset) % n
            fx = float(self._tp[future_idx][0]) - self.x
            fy = float(self._tp[future_idx][1]) - self.y
            target_angle = math.atan2(fy, fx)
            e = target_angle - self.angle
            return (e + math.pi) % (2 * math.pi) - math.pi

        return err_to(5), err_to(15), err_to(30)

    # ------------------------------------------------------------------ obs

    # builds the 12-element observation vector; all values clipped to [-1, 1]
    def _get_obs(self) -> np.ndarray:
        dist, signed_lat, heading_err = self._track_info()
        idx = self._nearest_track_idx()
        err_near, err_mid, err_far = self._lookahead_errors(idx)

        return np.array([
            self.speed / self.MAX_SPEED,
            math.sin(self.angle),
            math.cos(self.angle),
            np.clip(signed_lat / self.TRACK_WIDTH, -1.0, 1.0),
            math.sin(heading_err),
            math.cos(heading_err),
            math.sin(err_near),  math.cos(err_near),
            math.sin(err_mid),   math.cos(err_mid),
            math.sin(err_far),   math.cos(err_far),
        ], dtype=np.float32)

    # ------------------------------------------------------------------ step

    # advances the simulation one frame: applies action, updates physics,
    # computes reward, checks termination, and handles lap timing
    def step(self, action):
        steer, throttle = action

        if steer == 0:
            self.angle -= self.STEER_RATE
        elif steer == 2:
            self.angle += self.STEER_RATE

        if throttle == 1:
            self.speed = min(self.speed + self.ACCEL, self.MAX_SPEED)
        elif throttle == 2:
            self.speed = max(self.speed - self.BRAKE, 0.0)

        self.speed = max(self.speed - self.FRICTION, 0.0)
        self.x    += self.speed * math.cos(self.angle)
        self.y    += self.speed * math.sin(self.angle)
        self._step_count += 1

        self.current_lap_time = self.stopwatch.get_current_lap_time(self._step_count)

        dist, signed_lat, heading_err = self._track_info()

        curr_idx = self._nearest_track_idx()
        err_near, err_mid, err_far = self._lookahead_errors(curr_idx)

        # ---- reward ----
        reward    = 0.0
        alignment = math.cos(heading_err)

        # waypoints advanced this step; large jumps (e.g. wrap-around backwards) are ignored
        n            = len(self._tp)
        raw_progress = (curr_idx - self._last_track_idx) % n
        if raw_progress > n // 2:
            raw_progress = 0
        self._last_track_idx = curr_idx

        if raw_progress > 0 and alignment > 0.3:
            reward += raw_progress * 1.5
        elif raw_progress > 0 and self.speed <= 1.5:
            reward -= 0.1   # penalise crawling forward
        else:
            reward -= 0.05  # no progress made

        # wrong-way penalty
        if abs(heading_err) > math.pi / 2:
            reward -= 0.5

        # speed × alignment bonus
        reward += self.speed * alignment * 0.10

        # stall penalty
        if self.speed < 1.0:
            reward -= 0.15

        # quadratic edge penalty with a steeper cliff past 70% of track width
        edge_fraction = abs(signed_lat) / self.TRACK_WIDTH
        reward -= edge_fraction ** 2 * 0.15
        if edge_fraction > 0.7:
            reward -= (edge_fraction - 0.7) * 3.0

        # flat per-step time cost to encourage faster laps
        reward -= 0.02

        # reward steering toward the corner, penalise steering away
        if abs(err_near) > 0.2:
            steer_dir  =  1 if steer == 2 else -1 if steer == 0 else 0
            corner_dir =  1 if err_near > 0 else -1
            if steer_dir == corner_dir:
                reward += 0.5
            elif steer_dir == -corner_dir:
                reward -= 0.3

        # reward braking into sharp corners; penalise accelerating into them
        corner_sharpness = abs(err_near)
        if corner_sharpness > 0.4 and throttle == 2:
            reward += 0.2
        elif corner_sharpness > 0.4 and throttle == 1:
            reward -= 0.15

        # ---- termination ----
        terminated = False
        if dist > self.TRACK_WIDTH + self.OFF_ROAD_MARGIN:
            reward    -= 5.0 + self.speed * 1.5
            terminated = True

        truncated = self._step_count >= self.MAX_STEPS

        # lap completion: guard against triggering on the very first step
        if not terminated and not truncated and self._is_at_start() and self.current_lap_time > 3.0:
            self.stopwatch.update_best_time(self.current_lap_time)
            self.best_lap_time = self.stopwatch.get_best_time()
            self.stopwatch.start_lap(self._step_count)

        self.done = terminated
        return self._get_obs(), reward, terminated, truncated, {}

    # returns true when the car is near the start line and roughly facing forward
    def _is_at_start(self) -> bool:
        start = self._tp[10]
        dist  = math.hypot(self.x - start[0], self.y - start[1])
        return dist < self.TRACK_WIDTH and math.cos(self.angle) > 0.7

    # ------------------------------------------------------------------ render

    # draws the track, car, and HUD to the pygame window each frame.
    # initialises pygame and loads the car sprite on first call.
    def render(self):
        if self.render_mode != "human":
            return

        if self.screen is None:
            pygame.init()
            pygame.display.set_caption("RL Car - Training")
            self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
            self.clock  = pygame.time.Clock()
            try:
                img = pygame.image.load("car.png").convert_alpha()
                self.car_image = pygame.transform.scale(img, (32, 18))
            except Exception:
                self.car_image = None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.done = True

        self.screen.fill((34, 85, 34))

        def ws(pt):
            return (int(float(pt[0])), int(float(pt[1])))

        screen_pts = [ws(p) for p in self._tp]
        border_r   = int(self.TRACK_WIDTH + 7)
        tarmac_r   = int(self.TRACK_WIDTH)
        closed_pts = screen_pts + [screen_pts[0]]

        # draw border (slightly wider) then tarmac on top
        for i in range(len(closed_pts) - 1):
            pygame.draw.line(self.screen, (50, 50, 50),
                             closed_pts[i], closed_pts[i+1], border_r * 2)
        for pt in closed_pts:
            pygame.draw.circle(self.screen, (50, 50, 50), pt, border_r)

        for i in range(len(closed_pts) - 1):
            pygame.draw.line(self.screen, (90, 90, 90),
                             closed_pts[i], closed_pts[i+1], tarmac_r * 2)
        for pt in closed_pts:
            pygame.draw.circle(self.screen, (90, 90, 90), pt, tarmac_r)

        # centre dashes — walk each segment accumulating dash/gap phase
        DASH = 18;  GAP = 14;  dist_acc = 0.0;  drawing = True
        for i in range(len(closed_pts) - 1):
            x0, y0 = closed_pts[i];  x1, y1 = closed_pts[i+1]
            seg_len = math.hypot(x1-x0, y1-y0)
            if seg_len == 0: continue
            walked = 0.0
            while walked < seg_len:
                phase = DASH if drawing else GAP
                step  = min(phase - dist_acc, seg_len - walked)
                if drawing:
                    t0 = walked / seg_len;  t1 = (walked + step) / seg_len
                    pygame.draw.line(self.screen, (210, 210, 210),
                                     (int(x0+(x1-x0)*t0), int(y0+(y1-y0)*t0)),
                                     (int(x0+(x1-x0)*t1), int(y0+(y1-y0)*t1)), 2)
                walked += step;  dist_acc += step
                if dist_acc >= phase:
                    dist_acc = 0.0;  drawing = not drawing

        # start/finish checkerboard across track width
        p0 = self._tp[10];  p1 = self._tp[11]
        dx = float(p1[0]-p0[0]);  dy = float(p1[1]-p0[1])
        length = math.hypot(dx, dy)
        if length > 0:
            tx = dx/length;  ty = dy/length;  nx = -ty;  ny = tx
            sq = 8;  cols = int(self.TRACK_WIDTH * 2 // sq)
            lx = p0[0] + nx * self.TRACK_WIDTH
            ly = p0[1] + ny * self.TRACK_WIDTH
            for row in range(2):
                for col in range(cols):
                    color = (240,240,240) if (row+col)%2==0 else (20,20,20)
                    bx = lx - nx*(col*sq) + tx*(row*sq)
                    by = ly - ny*(col*sq) + ty*(row*sq)
                    pts = [ws((bx,by)),
                           ws((bx-nx*sq, by-ny*sq)),
                           ws((bx-nx*sq+tx*sq, by-ny*sq+ty*sq)),
                           ws((bx+tx*sq, by+ty*sq))]
                    pygame.draw.polygon(self.screen, color, pts)

        # draw car; fall back to a coloured rectangle if no image is loaded
        car_sx, car_sy = ws((self.x, self.y))
        if self.car_image:
            rot = pygame.transform.rotate(self.car_image, -math.degrees(self.angle))
        else:
            surf = pygame.Surface((26, 14), pygame.SRCALPHA)
            surf.fill((220, 30, 30))
            pygame.draw.rect(surf, (180, 220, 255), (16, 2, 7, 10))
            rot = pygame.transform.rotate(surf, -math.degrees(self.angle))
        self.screen.blit(rot, (car_sx - rot.get_width()//2, car_sy - rot.get_height()//2))

        # --- DRAW LOOKAHEAD DOTS ---
        idx = self._nearest_track_idx()
        n = len(self._tp)
        # Offsets 5 (Red), 15 (Yellow), 30 (Green)
        for offset, color in zip([5, 15, 30], [(255, 50, 50), (255, 255, 50), (50, 255, 50)]):
            target_idx = (idx + offset) % n
            tx, ty = self._tp[target_idx]
            pt = ws((tx, ty))
            pygame.draw.circle(self.screen, color, pt, 6)
            pygame.draw.circle(self.screen, (0, 0, 0), pt, 6, 1) # Outline for visibility

        # --- HUD ---
        if not hasattr(self, '_font'):
            self._font = pygame.font.SysFont("monospace", 16, bold=True)
        best_str = f"{self.best_lap_time:.2f}" if self.best_lap_time != float('inf') else "N/A"
        
        self.screen.blit(self._font.render(
            f"speed: {self.speed:.1f}  step: {self._step_count}", True, (240,240,240)), (10,10))
        self.screen.blit(self._font.render(
            f"Lap Time: {self.current_lap_time:.2f}", True, (240,240,240)), (10,30))
        self.screen.blit(self._font.render(
            f"Best Lap: {best_str}", True, (240,240,240)), (10,50))
        self.screen.blit(self._font.render(
            f"Waypoint: {idx} / {n}", True, (0,200,255)), (10,70))

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None