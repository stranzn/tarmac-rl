class Stopwatch:
    def __init__(self, fps: int = 60):
        self.fps = fps
        self.start_step = 0
        self.best_time = float('inf') 

    def start_lap(self, current_step: int):
        """start a new lap using the current simulation step."""
        self.start_step = current_step

    def get_current_lap_time(self, current_step: int) -> float:
        """calculate lap time based on how many simulation steps have passed."""
        steps_taken = current_step - self.start_step
        return steps_taken / self.fps

    def update_best_time(self, lap_time: float):
        """update best lap time if this one is faster."""
        if 0 < lap_time < self.best_time:
            self.best_time = lap_time

    def get_best_time(self) -> float:
        """return best time or infinity if none yet."""
        return self.best_time