import pygame
import subprocess
import sys

def run_script_and_exit(cmd_list):
    """Closes the menu window then launches the selected script in a new process."""
    print(f"Launching: {' '.join(cmd_list)}")
    subprocess.Popen(cmd_list)  # fire and forget — don't wait for it to finish
    pygame.quit()
    sys.exit()

def main_menu():
    pygame.init()
    width, height = 600, 500
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Tarmac RL - Main Menu")

    font_title    = pygame.font.SysFont("monospace", 42, bold=True)
    font_subtitle = pygame.font.SysFont("monospace", 14)
    font_btn      = pygame.font.SysFont("monospace", 22, bold=True)

    buttons = [
        {
            "label":    "Train Model (GUI)",
            "subtitle": "trains with live render window",
            "cmd":      [sys.executable, "train.py", "--gui"],
            "rect":     pygame.Rect(100, 130, 400, 55),
        },
        {
            "label":    "Train Model (Headless)",
            "subtitle": "faster — no render window",
            "cmd":      [sys.executable, "train.py"],
            "rect":     pygame.Rect(100, 205, 400, 55),
        },
        {
            "label":    "Test Trained Model",
            "subtitle": "watch car_ppo drive the track",
            "cmd":      [sys.executable, "test.py"],
            "rect":     pygame.Rect(100, 280, 400, 55),
        },
        {
            "label":    "Watch Ghost Replays",
            "subtitle": "all training snapshots at once",
            "cmd":      [sys.executable, "replay.py"],
            "rect":     pygame.Rect(100, 355, 400, 55),
        },
        {
            "label":    "Quit",
            "subtitle": "",
            "cmd":      "quit",
            "rect":     pygame.Rect(100, 430, 400, 45),
        },
    ]

    clock = pygame.time.Clock()

    while True:
        screen.fill((34, 40, 49))

        # Title
        title_surf = font_title.render("Tarmac RL", True, (238, 238, 238))
        screen.blit(title_surf, (width // 2 - title_surf.get_width() // 2, 40))

        # Subtitle
        sub_surf = font_subtitle.render("reinforcement learning car simulation", True, (120, 120, 130))
        screen.blit(sub_surf, (width // 2 - sub_surf.get_width() // 2, 90))

        mx, my = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for btn in buttons:
                    if btn["rect"].collidepoint((mx, my)):
                        if btn["cmd"] == "quit":
                            pygame.quit()
                            sys.exit()
                        else:
                            run_script_and_exit(btn["cmd"])

        # Draw buttons
        for btn in buttons:
            is_hovered = btn["rect"].collidepoint((mx, my))
            bg_color   = (0, 173, 181)   if is_hovered else (57, 62, 70)
            txt_color  = (255, 255, 255) if is_hovered else (238, 238, 238)
            sub_color  = (255, 255, 255) if is_hovered else (130, 130, 140)

            pygame.draw.rect(screen, bg_color,        btn["rect"], border_radius=10)
            pygame.draw.rect(screen, (238, 238, 238), btn["rect"], 2, border_radius=10)

            # Main label — shift up slightly if there's a subtitle
            has_sub  = bool(btn["subtitle"])
            label_y  = btn["rect"].centery - (10 if has_sub else 0) - font_btn.get_height() // 2
            lbl_surf = font_btn.render(btn["label"], True, txt_color)
            screen.blit(lbl_surf, (btn["rect"].centerx - lbl_surf.get_width() // 2, label_y))

            # Subtitle
            if has_sub:
                sub_surf = font_subtitle.render(btn["subtitle"], True, sub_color)
                screen.blit(sub_surf, (btn["rect"].centerx - sub_surf.get_width() // 2,
                                       label_y + font_btn.get_height() + 2))

        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main_menu()