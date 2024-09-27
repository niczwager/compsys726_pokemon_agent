from functools import cached_property

import numpy as np
from pyboy.utils import WindowEvent

from pyboy_environment.environments.pokemon.pokemon_environment import (
    PokemonEnvironment,
)
from pyboy_environment.environments.pokemon import pokemon_constants as pkc

import cv2

from sklearn.neighbors import KNeighborsClassifier

import hnswlib


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

        self.labels = []
        self.num_discovered_screens = 0

        # Initialize the hnswlib index
        #self.knn_index = hnswlib.Index(space='l2', dim=144*160)  
        self.knn_index = hnswlib.Index(space='l2', dim=101*101) # 100x100 frame size = 10,000 dimensions
        self.knn_index.init_index(max_elements=200000, ef_construction=200, M=16)  # Adjust max_elements as needed
        self.similar_frame_dist = 0.5  # Similarity threshold for adding new frames

        # Track if the index is empty
        self.index_initialized = False
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

        #pixels = cv2.cvtColor(pixels, cv2.COLOR_RGBA2GRAY)

        # Extracts AND converts image to GRAYSCALE from RGBA (4 channels)
        center_pixels = self.extract_center_pixels(pixels, size=100)

        #print(center_pixels.shape)

        #self.display_window(center_pixels)

        # Flatten the grayscale 100x100 frame into a 1D vector (for kNN)
        frame_vector = center_pixels.flatten()

        #print("Frame vector size bitchhhhh: ", frame_vector.shape)
        #self.inf_loop()

        reward = 0

        # Check if the current frame is new by querying the hnswlib model
        if self.is_new_screen(frame_vector, 0.75):
            # Add the frame to the hnswlib index
            self.update_frame_knn_index(frame_vector)
            reward += 1  # Reward for discovering a new frame

        return reward
        return new_state["badges"] - self.prior_game_stats["badges"]
    
    def is_new_screen(self, frame_vector, threshold=0.5):
        """
        Compares the current frame vector to all previously discovered frames using hnswlib.
        Args:
            frame_vector: The flattened 100x100 grayscale frame (numpy array).
            threshold: The distance threshold for determining if a frame is new (higher = more different).
        Returns:
            bool: True if the current frame is new, False otherwise.
        """
        # If no discovered frames exist, this is the first frame
        if not self.index_initialized:
            return True

        # Query the hnswlib index for the nearest neighbor
        labels, distances = self.knn_index.knn_query(frame_vector, k=1)

        # Normalize the distance based on the vector size
        normalized_distance = distances[0][0] / len(frame_vector)

        # If the distance is above the threshold, it is a new frame
        return normalized_distance >= threshold
    
    def update_frame_knn_index(self, frame_vec):
        """
        Updates the hnswlib index with the new frame.
        Args:
            frame_vec: The flattened frame vector to be added.
        """
        if not self.index_initialized:
            # Initialize the index by adding the first frame
            self.knn_index.add_items(frame_vec, np.array([self.knn_index.get_current_count()]))
            self.index_initialized = True
        else:
            # Directly add the frame to the index without checking threshold again
            self.knn_index.add_items(frame_vec, np.array([self.knn_index.get_current_count()]))

        # Increment the discovered screens count and add to labels
        self.num_discovered_screens += 1
        self.labels.append(self.num_discovered_screens)

    def _check_if_done(self, game_stats: dict[str, any]) -> bool:
        # Setting done to true if agent beats first gym (temporary)
        return game_stats["badges"] > self.prior_game_stats["badges"]

    def _check_if_truncated(self, game_stats: dict) -> bool:
        # Implement your truncation check logic here

        # Maybe if we run out of pokeballs...? or a max step count
        return self.steps >= 1000
