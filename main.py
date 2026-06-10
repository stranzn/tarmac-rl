import pygame
import subprocess
import sys

def run_script_and_exit(cmd_list):
    """Closes the menu and launches the selected script."""
    pygame.quit()
    print(f"Launching: {' '.join(cmd_list)}")
    subprocess.run(cmd_list)
    sys.exit()

def main_menu():
    pygame.init()
    width, height = 600, 450
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Tarmac RL - Main Menu")
    
    font_title = pygame.font.SysFont("monospace", 42, bold=True)
    font_btn = pygame.font.SysFont("monospace", 22, bold=True)

    buttons = [
        {"label": "Train Model (GUI)", "cmd": [sys.executable, "train.py", "--gui"], "rect": pygame.Rect(100, 120, 400, 50)},
        {"label": "Train Model (Headless)", "cmd": [sys.executable, "train.py"], "rect": pygame.Rect(100, 190, 400, 50)},
        {"label": "Test Trained Model", "cmd": [sys.executable, "test.py"], "rect": pygame.Rect(100, 260, 400, 50)},
        {"label": "Quit", "cmd": "quit", "rect": pygame.Rect(100, 330, 400, 50)}
    ]

    clock = pygame.time.Clock()

    while True:
        screen.fill((34, 40, 49)) # Dark sleek background
        
        # Title
        title_surf = font_title.render("Tarmac RL", True, (238, 238, 238))
        screen.blit(title_surf, (width//2 - title_surf.get_width()//2, 40))

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

        # Draw Buttons
        for btn in buttons:
            is_hovered = btn["rect"].collidepoint((mx, my))
            color = (0, 173, 181) if is_hovered else (57, 62, 70)
            text_color = (255, 255, 255) if is_hovered else (238, 238, 238)
            
            pygame.draw.rect(screen, color, btn["rect"], border_radius=10)
            pygame.draw.rect(screen, (238, 238, 238), btn["rect"], 2, border_radius=10)
            
            lbl_surf = font_btn.render(btn["label"], True, text_color)
            screen.blit(lbl_surf, (btn["rect"].centerx - lbl_surf.get_width()//2, 
                                   btn["rect"].centery - lbl_surf.get_height()//2))

        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main_menu()