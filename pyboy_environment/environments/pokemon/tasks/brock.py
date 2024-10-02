from functools import cached_property

import numpy as np
from pyboy.utils import WindowEvent

from pyboy_environment.environments.pokemon.pokemon_environment import (
    PokemonEnvironment,
)
from pyboy_environment.environments.pokemon import pokemon_constants as pkc

import cv2

import hnswlib

from skimage.transform import resize

'''
TODO:
- get this fucking cunt training
- the current issue is the animated section
- play around with what image size can be used that it trains and leaves out sufficient detail so the animation doesn't trip a difference
    - rectangular section cut off?
    - direct sprite find?
    - looks like the issue is the NPC - this is realtively constant so is there a way to filter this out?
'''


class PokemonBrock(PokemonEnvironment):
    def __init__(
        self,
        act_freq: int,
        emulation_speed: int = 0,
        headless: bool = False,
        frame_stacks: int = 3  # Added
    ) -> None:

        valid_actions: list[WindowEvent] = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
            #WindowEvent.PRESS_BUTTON_START,
        ]

        release_button: list[WindowEvent] = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
            #WindowEvent.RELEASE_BUTTON_START,
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

        self.original_x = 144
        self.original_y = 160

        self.kernel_size = (4,4)

        self.downsize_x = 144 // 2
        self.downsize_y = 160 // 2

        self.frame_stacks = frame_stacks  # Frame stacking for internal use
        self.recent_frames = np.zeros((self.frame_stacks, 42, 42, 3), dtype=np.uint8)  # Store stacked frames

        # Initialize the hnswlib index to use single frames (42x42x3) 
        self.knn_index = hnswlib.Index(space='l2', dim=144*160*3)  # Single frame dimension
        self.knn_index.init_index(max_elements=20000, ef_construction=100, M=16)

        # Track if the index is empty
        self.index_initialized = False
        self.prev_frame = None

        self.prev_x, self.prev_y = None, None

        self.prev_frame = None

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
    
    '''
    --------------------------------
    ADDED FUNCTIONS
    --------------------------------
    '''
    def extract_center_pixels(self, pixels, size=25):
        height, width, channels = pixels.shape
        assert height == 144 and width == 160 and channels == 4, "Unexpected screen dimensions."

        # Convert the RGBA image to grayscale
        grayscale_pixels = cv2.cvtColor(pixels, cv2.COLOR_RGBA2RGB)

        # Compute the center
        center_y, center_x = height // 2, width // 2
        half_size = size // 2

        # Slice the center 25x25 region (grayscale, so single channel)
        return grayscale_pixels[center_y - half_size:center_y + half_size + 1, 
                                center_x - half_size:center_x + half_size + 1]

    def extract_and_resize_pixels(self, pixels, target_size=(42, 42)):
        """
        Extracts the entire game screen and resizes it to 42x42 pixels in RGB.
        Args:
            pixels: The original screen's pixel array (expected to be 144x160x4).
            target_size: The target size for resizing (default is (42, 42)).
        Returns:
            Flattened numpy array of size (target_size[0] * target_size[1] * 3).
        """
        height, width, channels = pixels.shape
        assert height == 144 and width == 160 and channels == 4, "Unexpected screen dimensions."

        # Convert from RGBA to RGB (dropping the alpha channel)
        rgb_pixels = cv2.cvtColor(pixels, cv2.COLOR_RGBA2RGB)

        # Resize the entire image to the target size (42x42)
        resized_pixels = resize(rgb_pixels, target_size, anti_aliasing=True, preserve_range=True)

        # Convert the resized image back to uint8 (since resize returns float64)
        resized_pixels = resized_pixels.astype(np.uint8)

        # Return the flattened 42x42x3 array (RGB)
        return resized_pixels.flatten()

    def is_new_screen(self, frame_vector, threshold):
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
        #normalized_distance = distances[0] / len(frame_vector)

        # If the distance is above the threshold, it is a new frame
        return distances[0][0] > threshold
    
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

    def stack_frames(self, new_frame):
        """
        Update the frame stack by adding the new frame and removing the oldest.
        Args:
            new_frame: The newly captured frame (42x42x3).
        Returns:
            A flattened version of the stacked frames (42x42x3*self.frame_stacks).
        """
        # Reshape the new frame back into (42, 42, 3)
        reshaped_frame = new_frame.reshape(42, 42, 3)

        # Shift frames and add the new frame
        self.recent_frames = np.roll(self.recent_frames, 1, axis=0)
        self.recent_frames[0] = reshaped_frame

        # Return the stacked frames as a flattened array
        return self.recent_frames.flatten()

    '''
    --------------------------------
    ORIGINAL FUNCTIONS
    --------------------------------
    '''
    def _get_state(self) -> np.ndarray:
        # Implement your state retrieval logic here
        game_stats = self._generate_game_stats()

        # print(game_stats["location"]["x"])
        return [game_stats["badges"]]

    def _calculate_reward(self, new_state: dict) -> float:
        # Implement your reward calculation logic here

        '''
        Potential ideas:
        - small negative reward 
        
        '''

        # 500_000 -> trains until NPCs
        # 600_000 -> trains until NPCs
        # 800_000 -> doesn't train
        # 700_000 -> trains until NPCs
        # 750_000 -> trains until NPCs
        # 775_000 -> trains until NPCs
        # 787_500 -> doesn't train
        # 781_250 -> doesn't train
        # 778_125 -> trains until NPCs
        # 779_687.5 -> trains until NPCs
        # 780_500.0 -> doesn't train
        # 780_000 -> trains until NPCs
        # 780_250 -> trains until NPCs
        # 780_375 -> doesn't train
        # 780_312.5 -> doesn't train
        # 780_281.25 -> trains until NPCs
        # 780_296.875 -> trains until NPCs
        # 780_300 -> doesn't train
        # 780_298 -> trains until NPCs
        threshold = 780_299

        # LARGER THRESHOLD = FRAMES ARE REQUIRED TO BE MORE DIFFERENT
        # 0.35 -> trains 2 just reach NPCs
        # 0.4 -> doesn't train

        similarity_threshold = 0.375

        x_pos = new_state["location"]["x"]
        y_pos = new_state["location"]["y"]
        map = new_state["location"]["map_id"]

        # Retireve raw pixel values of current screen
        pixels = self.pyboy.screen.ndarray
        resized_frame = self.extract_and_resize_pixels(pixels)  # Resize to 42x42 RGB

        pixels_RGB = cv2.cvtColor(pixels, cv2.COLOR_RGBA2RGB)
        pixels_RGB = pixels_RGB.flatten()

        # Stack the new frame with recent frames
        #frame_stack_vector = self.stack_frames(resized_frame)

        reward = 0

        if self.is_new_screen(pixels_RGB, threshold):
            self.update_frame_knn_index(pixels_RGB)
            reward += 1

        #if self.prev_x == x_pos and self.prev_y == y_pos:
            #reward -= 0.1

        '''
        CHECKING PREVIOUS FRAME TO SEE IF IDENTICAL - DOESN'T SEEM TO WORK



          # Check if the current frame is new by querying the hnswlib model
        if self.is_new_screen(resized_frame, threshold):
            # After passing KNN, compare the current frame with the previous frame for similarity
            if self.prev_frame is not None:
                # IF FRAMES ARE SIMILAR YOU WOULD EXPECT A LOW DIFF
                diff = np.mean(self.prev_frame != pixels_RGB)
                
                # If frames are sufficiently different, give the reward
                if diff > similarity_threshold:
                    self.update_frame_knn_index(resized_frame)
                    reward += 1
                else:
                    pass
                    #print('KNN reached, but direct pixel match not reached')
                    #print(diff)
            else:
                # If there's no previous frame, just give the reward
                self.update_frame_knn_index(resized_frame)
                reward += 1
        '''

        self.prev_x, self.prev_y = x_pos, y_pos
        self.prev_frame = pixels_RGB
        

        return reward
        return new_state["badges"] - self.prior_game_stats["badges"]

    def _check_if_done(self, game_stats: dict[str, any]) -> bool:
        # Setting done to true if agent beats first gym (temporary)
        return game_stats["badges"] > self.prior_game_stats["badges"]

    def _check_if_truncated(self, game_stats: dict) -> bool:
        # Implement your truncation check logic here

        # Maybe if we run out of pokeballs...? or a max step count
        return self.steps >= 1000
