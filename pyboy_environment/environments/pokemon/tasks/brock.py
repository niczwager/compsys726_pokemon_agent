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

        # Initialize the hnswlib index to use single frames (42x42x3) 
        self.knn_index = hnswlib.Index(space='l2', dim=144*(160//2)*3)  # Single frame dimension
        self.knn_index.init_index(max_elements=20000, ef_construction=100, M=16)

        # Track if the index is empty
        self.index_initialized = False
        self.prev_frame = None

        self.prev_x, self.prev_y = None, None

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

    def calculate_frame_difference(self, current_frame, prev_frame):
        """Calculate the absolute difference between the current and previous frames."""
        return np.abs(current_frame - prev_frame)  # This preserves the original shape
    
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
        CURRENT IDEA:
        - only training on right half of the right image to see if that helps with NPC problem
        '''
        # 5000 -> doesn't train
        # 1000 -> doesn't train
        # 100 -> doesn't train
        # 10 -> doesn't train
        threshold = 100_000

        x_pos = new_state["location"]["x"]
        y_pos = new_state["location"]["y"]
        map = new_state["location"]["map_id"]

        # Retireve raw pixel values of current screen
        pixels = self.pyboy.screen.ndarray
        #resized_frame = self.extract_and_resize_pixels(pixels, (100, 100))  
        #resized_frame = self.extract_center_box_pixels(pixels, (100, 100))

        pixels_RGB = cv2.cvtColor(pixels, cv2.COLOR_RGBA2RGB)
        #pixels_RGB = pixels_RGB.flatten()

        # Extract the right half of the screen (slice only the right half, adjust as needed)
        height, width, _ = pixels_RGB.shape
        right_half = pixels_RGB[:, width // 2:]

        # Flatten the right half
        right_half_flattened = right_half.flatten()
        
        reward = 0

        if self.prev_frame is not None:
            prev_right_half = self.prev_frame[:, width // 2:]
            prev_right_half_flattened = prev_right_half.flatten()

            # Calculate the difference between the right halves
            diff = self.calculate_frame_difference(right_half_flattened, prev_right_half_flattened)

            if self.is_new_screen(diff, threshold):
                self.update_frame_knn_index(diff)
                #self.display_window([diff.reshape(144,160//2,3), right_half_flattened.reshape(144,160//2,3), prev_right_half_flattened.reshape(144,160//2,3)])
                reward += 1

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
