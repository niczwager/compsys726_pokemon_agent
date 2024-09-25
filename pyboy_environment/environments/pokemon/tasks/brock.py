from functools import cached_property

import numpy as np
from pyboy.utils import WindowEvent

from pyboy_environment.environments.pokemon.pokemon_environment import (
    PokemonEnvironment,
)
from pyboy_environment.environments.pokemon import pokemon_constants as pkc

import cv2


class PokemonBrock(PokemonEnvironment):
    def __init__(
        self,
        act_freq: int,
        emulation_speed: int = 0,
        headless: bool = False,
    ) -> None:

        valid_actions: list[WindowEvent] = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
            WindowEvent.PRESS_BUTTON_START,
        ]

        release_button: list[WindowEvent] = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
            WindowEvent.RELEASE_BUTTON_START,
        ]

        super().__init__(
            act_freq=act_freq,
            task="brock",
            init_name="has_pokedex.state",
            emulation_speed=emulation_speed,
            valid_actions=valid_actions,
            release_button=release_button,
            headless=headless,
        )

        self.discovered_maps = set()
        self.discovered_screens = []
    
    '''
    ---------------------
    USEFUL DEBUGGING CODE
    ---------------------
    '''
    def inf_loop(self):
        while True:
            pass
        
    def display_window(self, pixels):
        cv2.imshow("PNG", pixels)

        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

    def _get_state(self) -> np.ndarray:
        # Implement your state retrieval logic here
        game_stats = self._generate_game_stats()

        # print(game_stats["location"]["x"])
        return [game_stats["badges"]]
    
    def extract_center_pixels(self, pixels, size=25):
        height, width, channels = pixels.shape
        assert height == 144 and width == 160 and channels == 4, "Unexpected screen dimensions."

        # Convert the RGBA image to grayscale
        grayscale_pixels = cv2.cvtColor(pixels, cv2.COLOR_RGBA2GRAY)

        # Compute the center
        center_y, center_x = height // 2, width // 2
        half_size = size // 2

        # Slice the center 25x25 region (grayscale, so single channel)
        return grayscale_pixels[center_y - half_size:center_y + half_size + 1, 
                                center_x - half_size:center_x + half_size + 1]


    def _calculate_reward(self, new_state: dict) -> float:
        # Implement your reward calculation logic here

        x_pos = new_state["location"]["x"]
        y_pos = new_state["location"]["y"]
        map = new_state["location"]["map_id"]

        # Retireve raw pixel values of current screen
        pixels = self.pyboy.screen.ndarray

        # Extracts AND converts image to GRAYSCALE from RGBA (4 channels)
        center_pixels = self.extract_center_pixels(pixels, size=100)

        #self.display_window(center_pixels)

        reward = 0

        if self.is_new_screen(center_pixels):
            self.discovered_screens.append(center_pixels)
            reward += 1  # Reward for discovering a new frame
        else:
            reward -= 100 # Heavily affecting reward if no new frames found

        # If no new location is discovered, calculate the badge difference
        return reward
        return new_state["badges"] - self.prior_game_stats["badges"]
    
    def is_new_screen(self, current_frame, threshold=0.8):
        """
        Compares the current frame to all previously discovered frames and checks if at least
        `threshold` (80%) of the pixels are different.
        Args:
            current_frame: The current 100x100 screen frame (numpy array).
            threshold: The percentage difference required to consider the frame new.
        Returns:
            bool: True if the current frame is new, False otherwise.
        """
        if not self.discovered_screens:
            # If no discovered screens exist, this is the first screen
            return True

        total_pixels = current_frame.size  # Total number of pixels in the grayscale frame
        for discovered_frame in self.discovered_screens:
            # Calculate the number of different pixels
            num_different_pixels = np.sum(current_frame != discovered_frame)
            percent_different = num_different_pixels / total_pixels

            # If more than `threshold` percent pixels are different, consider it new
            if percent_different >= threshold:
                
                return True

        # If none of the previously discovered frames are sufficiently different, it's not new
        return False

    def _check_if_done(self, game_stats: dict[str, any]) -> bool:
        # Setting done to true if agent beats first gym (temporary)
        return game_stats["badges"] > self.prior_game_stats["badges"]

    def _check_if_truncated(self, game_stats: dict) -> bool:
        # Implement your truncation check logic here

        # Maybe if we run out of pokeballs...? or a max step count
        return self.steps >= 1000
