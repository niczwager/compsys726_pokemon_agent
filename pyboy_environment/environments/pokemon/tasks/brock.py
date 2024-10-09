
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

        # Track if the index is empty
        self.index_initialized = False
        self.prev_frame = None

        self.prev_x, self.prev_y = None, None

        # PYBOY VARIBLES
        self.levels_satisfied = False
        self.output_shape = (42,42,3)
        self.vec_dim = self.output_shape[0]*self.output_shape[1]*self.output_shape[2]
        self.base_explore = 0
        # 100_000 -> trains until NPCs
        # 105_000 -> doesn't train
        # 102_500 -> doesn't train
        # 100_500 -> doesn't train
        # 100_250 -> doesn't train
        self.similar_frame_dist = 100_000.0

        # Initialize the hnswlib index to use single frames (42x42x3) 
        self.knn_index = hnswlib.Index(space='l2', dim=self.vec_dim)  # Single frame dimension
        self.knn_index.init_index(max_elements=20000, ef_construction=100, M=16)

    '''
    ---------------------
    USEFUL DEBUGGING CODE
    ---------------------
    '''
    def inf_loop(self):
        while True:
            pass
        
    def display_window(self, screens):
        # Iterate through the list of screens and display each in its own window
        for i, pixels in enumerate(screens):
            window_name = f"Screen {i+1}"
            cv2.imshow(window_name, pixels)

        # Wait for a key press to close all windows
        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Close all windows
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
    
    def extract_center_box_pixels(self, pixels, box_size=(42, 42)):

        height, width, channels = pixels.shape
        assert height == 144 and width == 160 and channels == 4, "Unexpected screen dimensions."

        # Convert from RGBA to RGB (dropping the alpha channel)
        rgb_pixels = cv2.cvtColor(pixels, cv2.COLOR_RGBA2RGB)

        # Calculate the center of the image
        center_y, center_x = height // 2, width // 2
        box_half_height, box_half_width = box_size[0] // 2, box_size[1] // 2

        # Extract the 42x42 box from the center of the image
        center_box = rgb_pixels[center_y - box_half_height:center_y + box_half_height, 
                                center_x - box_half_width:center_x + box_half_width]
        
        #self.display_window(center_box)

        # Flatten the 42x42x3 array (RGB)
        return center_box.flatten()

    def is_new_screen(self, frame_vector, threshold):
        # If no discovered frames exist, this is the first frame
        if not self.index_initialized:
            return True

        # Query the hnswlib index for the nearest neighbor
        labels, distances = self.knn_index.knn_query(frame_vector, k=1)

        #print(distances[0][0])

        # If the distance is above the threshold, it is a new frame
        return distances[0][0] > threshold
    
    '''
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
    '''

    def update_frame_knn_index(self, frame_vec):
        
        '''
        if self.get_levels_sum() >= 22 and not self.levels_satisfied:
            self.levels_satisfied = True
            self.base_explore = self.knn_index.get_current_count()
            self.init_knn()
        '''

        if self.knn_index.get_current_count() == 0:
            # if index is empty add current frame
            self.knn_index.add_items(
                frame_vec, np.array([self.knn_index.get_current_count()])
            )

            return True
        else:
            # check for nearest frame and add if current 
            labels, distances = self.knn_index.knn_query(frame_vec, k = 1)
            if distances[0] > self.similar_frame_dist:
                self.knn_index.add_items(
                    frame_vec, np.array([self.knn_index.get_current_count()])
                )

                return True
        
        return False

    def calculate_frame_difference(self, current_frame, prev_frame):
        """Calculate the absolute difference between the current and previous frames."""
        return np.abs(current_frame - prev_frame)  # This preserves the original shape
    
    def is_box_identical(self, current_frame, prev_frame, box_size=20, offset=50):
        """
        Check if a box of size box_size offset to the right by offset pixels from the center is identical
        between the current and previous frames.
        """
        height, width, _ = current_frame.shape

        # Define the center of the image and offset to the right
        center_y = height // 2
        center_x = width // 2 + offset

        # Define the half size of the box
        box_half_size = box_size // 2

        # Extract the box region from the current and previous frames
        box_current = current_frame[
            center_y - box_half_size:center_y + box_half_size,
            center_x - box_half_size:center_x + box_half_size
        ]

        box_prev = prev_frame[
            center_y - box_half_size:center_y + box_half_size,
            center_x - box_half_size:center_x + box_half_size
        ]

        #self.display_window([box_current, box_prev, current_frame])

        # Check if the regions are identical
        return np.array_equal(box_current, box_prev)
    
    def resize_frame(self, current_frame):
        return (255*resize(current_frame, self.output_shape)).astype(np.uint8)
    
    def get_knn_reward(self):
        pre_rew = 0.004

        cur_size = self.knn_index.get_current_count()
        base = (self.base_explore if self.levels_satisfied else cur_size) * pre_rew

        return base 
    
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
        
        threshold = 12_500

        x_pos = new_state["location"]["x"]
        y_pos = new_state["location"]["y"]
        map = new_state["location"]["map_id"]

        # Retireve raw pixel values of current screen
        pixels = self.pyboy.screen.ndarray
        #resized_frame = self.extract_and_resize_pixels(pixels, (100, 100))  
        #resized_frame = self.extract_center_box_pixels(pixels, (100, 100))

        pixels = cv2.cvtColor(pixels, cv2.COLOR_RGBA2RGB)
        pixels = self.resize_frame(pixels)
        #pixels_RGB = pixels_RGB.flatten()
        
        reward = 0

        if self.update_frame_knn_index(pixels.flatten()):
            reward += self.get_knn_reward()
        
        '''
        if self.prev_x != x_pos or self.prev_y != y_pos:
            reward *= 10

        self.prev_x, self.prev_y = x_pos, y_pos
        '''
        
        return reward
        return new_state["badges"] - self.prior_game_stats["badges"]

    def _check_if_done(self, game_stats: dict[str, any]) -> bool:
        # Setting done to true if agent beats first gym (temporary)
        return game_stats["badges"] > self.prior_game_stats["badges"]

    def _check_if_truncated(self, game_stats: dict) -> bool:
        # Implement your truncation check logic here

        # Maybe if we run out of pokeballs...? or a max step count
        return self.steps >= 1000