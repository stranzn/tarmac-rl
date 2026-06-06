import numpy as np
import math

class TrackBuilder:
    """builds 2d track geometry by chaining straight and turn segments."""

    def __init__(self, density: float = 1.0):
        self.points = []
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_angle = 0.0
        self.density = density  # points per unit of arc/straight length

    # resets the builder state so a new track can be constructed from scratch
    def reset(self):
        self.points = []
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_angle = 0.0
        return self

    # appends points along a straight segment of the given distance
    def straight(self, distance: float):
        steps = max(1, int(distance * self.density))
        
        for i in range(1, steps + 1):
            t = i / steps
            self.points.append((
                self.current_x + distance * t * math.cos(self.current_angle),
                self.current_y + distance * t * math.sin(self.current_angle),
            ))
        self.current_x += distance * math.cos(self.current_angle)
        self.current_y += distance * math.sin(self.current_angle)
        return self

    # appends points along a circular arc; positive angle_deg turns left, negative turns right
    def turn(self, angle_deg: float, radius: float):
        angle_rad = math.radians(angle_deg)
        arc_length = abs(radius * angle_rad)
        steps = max(1, int(arc_length * self.density))

        # pivot centre sits perpendicular to current heading
        if angle_deg >= 0:
            perp = self.current_angle + math.pi / 2
        else:
            perp = self.current_angle - math.pi / 2
        cx = self.current_x + radius * math.cos(perp)
        cy = self.current_y + radius * math.sin(perp)
        start_arm = math.atan2(self.current_y - cy, self.current_x - cx)

        for i in range(1, steps + 1):
            t = i / steps
            arm = start_arm + angle_rad * t
            self.points.append((cx + radius * math.cos(arm),
                                cy + radius * math.sin(arm)))

        final_arm = start_arm + angle_rad
        self.current_x = cx + radius * math.cos(final_arm)
        self.current_y = cy + radius * math.sin(final_arm)
        self.current_angle += angle_rad
        return self

    # returns all accumulated points as a float32 array
    def get_points(self) -> np.ndarray:
        return np.array(self.points, dtype=np.float32)

    # a simple custom circuit for quick testing
    def build_custom_circuit(self):
        self.reset()
        self.straight(600)
        self.turn(45, 40).turn(-90, 40).turn(45, 40)
        self.straight(150).turn(90, 300)
        self.straight(200).turn(-45, 50).turn(90, 50).turn(-45, 50)
        self.straight(100).turn(45, 120).straight(50).turn(45, 120)
        self.straight(400)
        self.turn(-45, 80).turn(90, 80).turn(-45, 80)
        self.straight(300).turn(90, 150).turn(90, 300)
        
        return self.get_points()

    # rough approximation of the monza circuit.
    # agent was able to complete a lap within ~400 training iterations.
    # took a comedic amount of time to model.
    def build_monza_circuit(self):
        self.reset()
        self.straight(600)
        self.turn(90, 10).straight(100).turn(-90,50)
        self.turn(-50, 350)
        self.turn(45, 400)
        self.turn(90, 400)
        self.turn(10, 300)
        self.straight(400)
        self.turn(-60, 90)
        self.turn(50, 150).straight(200)
        self.turn(75, 150).straight(500)
        self.turn(100, 90).straight(50).turn(-30, 1000).straight(500)
        self.turn(-60, 30).straight(100).turn(40, 200)
        self.turn(-30, 100).straight(900)
        self.turn(180, 210).straight(741)
        
        return self.get_points()