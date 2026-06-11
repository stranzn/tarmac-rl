import pygame
import numpy as np
import math
import sys
from car_env import CarEnv

REPLAY_FILE = "replays.npy"
FPS         = 60

def colour_for(index: int, total: int) -> tuple:
    # red (early training) -> green (late training), interpolated per ghost
    t   = index / max(total - 1, 1)
    r   = int(220 * (1 - t))
    g   = int(220 * t)
    return (r, g, 40)

def draw_car(surface, x, y, angle, colour, alpha=180):
    # draw a simple coloured arrow representing a ghost car
    surf = pygame.Surface((26, 14), pygame.SRCALPHA)
    pygame.draw.polygon(surf, (*colour, alpha), [(0,2),(0,12),(20,12),(20,10),(26,7),(20,4),(20,2)])
    rotated = pygame.transform.rotate(surf, -math.degrees(angle))
    surface.blit(rotated, (int(x) - rotated.get_width()  // 2,
                           int(y) - rotated.get_height() // 2))

def main():
    # load replays
    try:
        replays = np.load(REPLAY_FILE, allow_pickle=True).tolist()
    except FileNotFoundError:
        print(f"No replay file found at '{REPLAY_FILE}'.")
        print("Train the model first — replays are recorded every 5k timesteps.")
        sys.exit(1)

    n = len(replays)
    if n == 0:
        print("Replay file is empty.")
        sys.exit(1)

    print(f"Loaded {n} replays. Playing all simultaneously...")
    print("Controls: SPACE = pause/resume | R = restart | ESC = quit")

    # spin up a headless env just to get the track geometr
    ref_env = CarEnv(render_mode=None)
    ref_env.reset()

    # pygame setup
    pygame.init()
    screen = pygame.display.set_mode((ref_env.SCREEN_WIDTH, ref_env.SCREEN_HEIGHT))
    pygame.display.set_caption("tarmac-rl — Ghost Replay")
    clock  = pygame.font.SysFont("monospace", 15)
    font   = pygame.font.SysFont("monospace", 15)

    # pre-compute colours for each ghost
    colours = [colour_for(i, n) for i in range(n)]

    # max frame count across all replays
    max_frames = max(len(r) for r in replays)

    frame    = 0
    paused   = False
    running  = True
    tk       = pygame.time.Clock()

    def ws(pt):
        return (int(float(pt[0])), int(float(pt[1])))

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    frame = 0

        if not paused:
            frame += 1
            if frame >= max_frames:
                frame = 0   # loop

        # draw track
        screen.fill((34, 85, 34))

        tp         = ref_env._tp
        screen_pts = [ws(p) for p in tp]
        border_r   = int(ref_env.TRACK_WIDTH + 7)
        tarmac_r   = int(ref_env.TRACK_WIDTH)
        closed_pts = screen_pts + [screen_pts[0]]

        for i in range(len(closed_pts) - 1):
            pygame.draw.line(screen, (50, 50, 50),
                             closed_pts[i], closed_pts[i+1], border_r * 2)
        for pt in closed_pts:
            pygame.draw.circle(screen, (50, 50, 50), pt, border_r)

        for i in range(len(closed_pts) - 1):
            pygame.draw.line(screen, (90, 90, 90),
                             closed_pts[i], closed_pts[i+1], tarmac_r * 2)
        for pt in closed_pts:
            pygame.draw.circle(screen, (90, 90, 90), pt, tarmac_r)

        # Centre dashes
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
                    t0 = walked/seg_len;  t1 = (walked+step)/seg_len
                    pygame.draw.line(screen, (210,210,210),
                                     (int(x0+(x1-x0)*t0), int(y0+(y1-y0)*t0)),
                                     (int(x0+(x1-x0)*t1), int(y0+(y1-y0)*t1)), 2)
                walked += step;  dist_acc += step
                if dist_acc >= phase:
                    dist_acc = 0.0;  drawing = not drawing

        # draw all ghosts at current frame
        for i, replay in enumerate(replays):
            if frame < len(replay):
                x, y, angle = replay[frame]
                draw_car(screen, x, y, angle, colours[i], alpha=200)
            else:
                # Episode ended — draw ghost frozen at last known position
                x, y, angle = replay[-1]
                draw_car(screen, x, y, angle, colours[i], alpha=60)

        # legend
        legend_y = 10
        screen.blit(font.render(f"Ghosts: {n}  |  Frame: {frame}/{max_frames}  |  "
                                f"SPACE=pause  R=restart  ESC=quit",
                                True, (240, 240, 240)), (10, legend_y))

        # colour gradient legend
        bar_x = 10;  bar_y = 35;  bar_w = 200;  bar_h = 12
        for px in range(bar_w):
            t   = px / bar_w
            r   = int(220 * (1 - t));  g = int(220 * t)
            pygame.draw.line(screen, (r, g, 40), (bar_x+px, bar_y), (bar_x+px, bar_y+bar_h))
        pygame.draw.rect(screen, (200, 200, 200), (bar_x, bar_y, bar_w, bar_h), 1)
        screen.blit(font.render("early", True, (240,240,240)), (bar_x,        bar_y + bar_h + 2))
        screen.blit(font.render("late",  True, (240,240,240)), (bar_x+bar_w-28, bar_y + bar_h + 2))

        pygame.display.flip()
        tk.tick(FPS)

    ref_env.close()
    pygame.quit()

if __name__ == "__main__":
    main()