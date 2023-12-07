from vis_nav_game import Player, Action
import pygame
import cv2
import numpy as np
import sys
import random
from place_recognition import extract_sift_features, create_visual_dictionary, generate_feature_histograms, compare_histograms, process_image_and_find_best_match
import networkx as nx
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import redis
import json
import subprocess

ROTATE_VALUE = 2.415
MOVE_VALUE = 5

class KeyboardPlayerPyGame(Player):
    def __init__(self):
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None
        self.keymap = None
        self.images = []
        self.visual_dictionary = None
        self.histograms = None

        self.validation_img = None

        self.turn_count = 0

        self.poses = []
        self.self_validate = False
        self.target_index = -1
        self.graph = None
        self.target = None

        # Initialize the map data
        self.map_size = (1000, 1000, 3)  # Example size for a larger map
        self.map_data = np.zeros(self.map_size, dtype=np.uint8)
        self.direction = 0  # Represents the current angle in degrees
        self.map_scale = 4  # Each unit in the map_data will be a 4x4 pixel square in the OpenCV window

        
        self.player_position = (0,0)
        self.key_hold_state = {pygame.K_LEFT: False, pygame.K_RIGHT: False, pygame.K_UP: False, pygame.K_DOWN: False}
        self.key_hold_time = {pygame.K_LEFT: {'start':0, 'end':0}, pygame.K_RIGHT: {'start':0, 'end':0}, pygame.K_UP: {'start':0, 'end':0}, pygame.K_DOWN: {'start':0, 'end':0}}
        
        self.redis = redis.Redis(host='127.0.0.1', port=6379, db=0) 
        self.redis.flushall()
        super(KeyboardPlayerPyGame, self).__init__()

    def reset(self):
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None

        # Reset map data
        self.map_data.fill(0)
        self.direction = 0
        
        self.player_position = (0,0)

        pygame.init()

        self.keymap = {
            pygame.K_LEFT: Action.LEFT,
            pygame.K_RIGHT: Action.RIGHT,
            pygame.K_UP: Action.FORWARD,
            pygame.K_DOWN: Action.BACKWARD,
            pygame.K_SPACE: Action.CHECKIN,
            pygame.K_ESCAPE: Action.QUIT,
            pygame.K_p: 1,
            pygame.K_r: 1,
            pygame.K_t: 1
        }

    def act(self):
        if 0 < self.turn_count < 37:
            self.turn_count += 1
            return self.last_act
        else:
            self.turn_count = 0
            pygame.event.set_allowed(pygame.KEYDOWN)
            pygame.event.set_allowed(pygame.KEYUP)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    self.last_act = Action.QUIT
                    self.redis.flushall()
                    self.redis.flushdb()
                    self.redis.close()
                    return Action.QUIT

                if event.type == pygame.KEYDOWN:
                    if event.key in self.keymap:
                        if event.key == pygame.K_p:
                            self.pre_navigation_bypass()
                        elif event.key == pygame.K_r:
                            self.player_position = (0,0)
                        elif event.key == pygame.K_t:
                            self.direction = 0

                        else:
                            self.key_hold_state[event.key] = True
                            self.last_act |= self.keymap[event.key]
                            if self.keymap[event.key] == Action.LEFT:  
                                pygame.event.set_blocked(pygame.KEYDOWN)
                                pygame.event.set_blocked(pygame.KEYUP)
                                self.turn_count = 1
                                self.direction += 90
                            elif self.keymap[event.key] == Action.RIGHT:
                                pygame.event.set_blocked(pygame.KEYDOWN)
                                pygame.event.set_blocked(pygame.KEYUP)
                                self.turn_count = 1
                                self.direction += -90

                        
                    else:
                        
                        self.show_target_images()
                if event.type == pygame.KEYUP:
                    if event.key in self.keymap:
                        if event.key == pygame.K_p or event.key == pygame.K_r or event.key == pygame.K_t:
                            pass
                        else:
                            self.key_hold_state[event.key] = False
                            self.last_act ^= self.keymap[event.key]
        
        self.update_map_on_keypress()
        return self.last_act
    

    def show_target_images_default(self):
        targets = self.get_target_images()
        if targets is None or len(targets) <= 0:
            return
        hor1 = cv2.hconcat(targets[:2])
        hor2 = cv2.hconcat(targets[2:])
        concat_img = cv2.vconcat([hor1, hor2])

        w, h = concat_img.shape[:2]
        
        color = (0, 0, 0)

        concat_img = cv2.line(concat_img, (int(h/2), 0), (int(h/2), w), color, 2)
        concat_img = cv2.line(concat_img, (0, int(w/2)), (h, int(w/2)), color, 2)

        w_offset = 25
        h_offset = 10
        font = cv2.FONT_HERSHEY_SIMPLEX
        line = cv2.LINE_AA
        size = 0.75
        stroke = 1

        cv2.putText(concat_img, 'Front View', (h_offset, w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Left View', (int(h/2) + h_offset, w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Back View', (h_offset, int(w/2) + w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Right View', (int(h/2) + h_offset, int(w/2) + w_offset), font, size, color, stroke, line)

        cv2.imshow(f'KeyboardPlayer:target_images', concat_img)
        cv2.waitKey(1)

    def show_target_images(self):
        self.player_position = (0,0)
        self.direction = 0
        targets = self.get_target_images()
        if self.self_validate and len(targets) > 0:
            targets[0] = self.validation_img

        best_indexes = []
        
        for target in targets:

            if self.visual_dictionary is None:
                raise AttributeError("Dictionary was None")
            
            best_index = process_image_and_find_best_match(target,self.histograms,self.visual_dictionary)
            best_indexes.append(best_index)

        if targets is None or len(targets) == 0:
            return

        # Concatenate best match images in pairs horizontally and then vertically
        hor1 = cv2.hconcat([self.images[best_indexes[0][0]], self.images[best_indexes[1][0]]])
        hor2 = cv2.hconcat([self.images[best_indexes[2][0]], self.images[best_indexes[3][0]]])
        concat_img = cv2.vconcat([hor1, hor2])

        # Concatenate second best match images similarly as above
        hor1_second_best = cv2.hconcat([self.images[best_indexes[0][1]], self.images[best_indexes[1][1]]])
        hor2_second_best = cv2.hconcat([self.images[best_indexes[2][1]], self.images[best_indexes[3][1]]])
        concat_img_second_best = cv2.vconcat([hor1_second_best, hor2_second_best])

        # Concatenate target images similarly as above
        hor1_target = cv2.hconcat(targets[:2])
        hor2_target = cv2.hconcat(targets[2:])
        concat_img_target = cv2.vconcat([hor1_target, hor2_target])

        # Concatenate third best match images similarly as above
        hor1_third_best = cv2.hconcat([self.images[best_indexes[0][2]], self.images[best_indexes[1][2]]])
        hor2_third_best = cv2.hconcat([self.images[best_indexes[2][2]], self.images[best_indexes[3][2]]])
        concat_img_third_best = cv2.vconcat([hor1_third_best, hor2_third_best])

        # Get width and height for text placement before scaling
        w, h = concat_img_target.shape[:2]

        # Settings for text
        w_offset = 25
        h_offset = 10
        font = cv2.FONT_HERSHEY_SIMPLEX
        line = cv2.LINE_AA
        size = 0.75
        stroke = 1
        color = (0, 0, 0)

        # Scaling factor for the images
        scale_factor = 2
        text_scale_factor = 1.2

        # Resize images with scale factor
        concat_img = cv2.resize(concat_img, (0, 0), fx=scale_factor, fy=scale_factor)
        concat_img_second_best = cv2.resize(concat_img_second_best, (0, 0), fx=scale_factor, fy=scale_factor)
        concat_img_third_best = cv2.resize(concat_img_third_best, (0, 0), fx=scale_factor, fy=scale_factor)
        concat_img_target = cv2.resize(concat_img_target, (0, 0), fx=scale_factor, fy=scale_factor)


        # Settings for text after scaling
        font = cv2.FONT_HERSHEY_SIMPLEX
        line = cv2.LINE_AA
        color = (0, 0, 255)  # Red color for visibility

        # Scale offsets and stroke for the scaled image
        scaled_w_offset = int(w_offset * scale_factor)
        scaled_h_offset = int(h_offset * scale_factor)
        scaled_font_size = size * text_scale_factor
        scaled_stroke = int(stroke * scale_factor)

        # Calculate positions for text based on scaled image size
        position_front_view = (scaled_w_offset, scaled_h_offset)
        position_right_view = (int(concat_img_target.shape[1]/2) + scaled_w_offset, scaled_h_offset)
        position_back_view = (scaled_w_offset, int(concat_img_target.shape[0]/2) + scaled_h_offset)
        position_left_view = (int(concat_img_target.shape[1]/2) + scaled_w_offset, int(concat_img_target.shape[0]/2) + scaled_h_offset)

        # Place text for views on the target image
        cv2.putText(concat_img_target, 'Front View', position_front_view, font, scaled_font_size, color, scaled_stroke, line)
        cv2.putText(concat_img_target, 'Right View', position_right_view, font, scaled_font_size, color, scaled_stroke, line)
        cv2.putText(concat_img_target, 'Back View', position_back_view, font, scaled_font_size, color, scaled_stroke, line)
        cv2.putText(concat_img_target, 'Left View', position_left_view, font, scaled_font_size, color, scaled_stroke, line)

        # Now, apply the text with correct scaling and positioning
        for index in range(1, 5):  # Loop through the indexes
            for rank in range(3):  # Loop through the ranks: best, second best, third best
                # Choose the correct image based on rank
                if rank == 0:
                    image_to_draw = concat_img
                elif rank == 1:
                    image_to_draw = concat_img_second_best
                else:
                    image_to_draw = concat_img_third_best
                
                # Calculate the position for the text based on the image quadrant
                x_offset = h_offset if (index % 2) != 0 else int(w/2) + h_offset  # Left if 1 or 3, right if 2 or 4
                y_offset = w_offset if index < 3 else int(w/2) + w_offset  # Top if 1 or 2, bottom if 3 or 4

                # Draw the text with the scaled positions and sizes
                cv2.putText(image_to_draw, 
                            f'Selection: {3*(index-1) + rank + 1}\t\t', 
                            (x_offset * scale_factor, y_offset * scale_factor),  # Scaled offset
                            font, 
                            scaled_font_size, 
                            color, 
                            scaled_stroke, 
                            line)
        # Concatenate the images again for the final display
        top_row = cv2.hconcat([concat_img, concat_img_second_best])
        bottom_row = cv2.hconcat([concat_img_third_best, concat_img_target])

        # Create and resize window for display
        cv2.namedWindow('KeyboardPlayer:targets and recognized', cv2.WINDOW_NORMAL)  # Create a resizable window
        cv2.resizeWindow('KeyboardPlayer:targets and recognized', top_row.shape[1], top_row.shape[0])  # Set the window size

        # Display the image
        cv2.imshow('KeyboardPlayer:targets and recognized', cv2.vconcat([top_row, bottom_row]))
        cv2.waitKey(1)
        return best_indexes

    def set_target_images(self, images):
        super(KeyboardPlayerPyGame, self).set_target_images(images)
        self.target = self.show_target_images()
        
        if self.target is not None:
            self.graph = self.create_graph_from_poses()
            self.target = np.ravel(self.target)
            self.target_index = int(input(f"Enter the row index (between 0 and {len(self.target) - 1}): ")) - 1
            cv2.destroyAllWindows()
            self.show_target_images_default()
            cv2.moveWindow("KeyboardPlayer:target_images", 500, 200)
            

    def pre_navigation(self):
        print("pre_nav")
        targets = self.get_target_images()

    def pre_navigation_bypass(self) -> None:
        if len(self.images) != 0:
            if self.self_validate:
                index = random.randint(10,len(self.images)-10)
                self.validation_img = self.images[index]
                for i in range(-5,5):
                    del self.images[index + i]
                    del self.poses[index + i]

            print(f"\nFinding descriptors for {len(self.images)} images, with {len(self.poses)} possible poses")
            keypoints,descriptors = extract_sift_features(self.images)
            print(f"Creating dictionary for images")
            self.visual_dictionary = create_visual_dictionary(np.vstack(descriptors), num_clusters=100)
            print(f"Creating {len(self.images)} histograms")
            self.histograms = generate_feature_histograms(descriptors, self.visual_dictionary)   

    def find_targets(self):
        targets = self.get_target_images()
        for target in targets:
            best_indexes = process_image_and_find_best_match(target,self.histograms,self.visual_dictionary)
            # cv2.imshow("best target", self.images[best_indexes[0]])

    def see(self, fpv):
        if fpv is None or len(fpv.shape) < 3:
            return

        self.fpv = fpv

        if self.screen is None:
            h, w, _ = fpv.shape
            self.screen = pygame.display.set_mode((w, h))

        def convert_opencv_img_to_pygame(opencv_image):
            """
            Convert OpenCV images for Pygame.

            see https://blanktar.jp/blog/2016/01/pygame-draw-opencv-image.html
            """
            opencv_image = opencv_image[:, :, ::-1]  # BGR->RGB
            shape = opencv_image.shape[1::-1]  # (height,width,Number of colors) -> (width, height)
            pygame_image = pygame.image.frombuffer(opencv_image.tobytes(), shape, 'RGB')

            return pygame_image

        pygame.display.set_caption("KeyboardPlayer:fpv")
        self.images.append(fpv)
        self.poses.append(tuple([self.player_position[0],self.player_position[1],self.direction]))
        rgb = convert_opencv_img_to_pygame(fpv)
        self.screen.blit(rgb, (0, 0))
        if self.target_index > -1:
            # curr_map = self.draw_graph_with_target()
            # cv2.imshow("2D map", curr_map)
            # cv2.waitKey(1)
            self.update_redis_data()
        pygame.display.update()

    def update_map_on_keypress(self):

        self.direction %= 360
        move_x, move_y = 0, 0

        if self.key_hold_state[pygame.K_UP] or self.key_hold_state[pygame.K_DOWN]:
            move_amount = MOVE_VALUE if self.key_hold_state[pygame.K_UP] else -MOVE_VALUE
            move_x = move_amount * np.cos(np.deg2rad(self.direction))
            move_y = move_amount * np.sin(np.deg2rad(self.direction))
        self.player_position = (self.player_position[0] + move_x, self.player_position[1] + move_y)

        sys.stdout.write(f'\rX: {self.player_position[0]:.2f} Y:{self.player_position[1]:.2f} W: {self.direction:.2f}')
        sys.stdout.flush()


    def create_graph_from_poses(self, threshold=25):
        if self.poses is None or len(self.poses) == 0:
            return None
        """
        Create a graph from a list of poses where each pose is [x, y, rotation].
        Connect nodes if their Euclidean distance is less than a given threshold.
        """
        # Create an empty graph
        graph = nx.Graph()

        # List to keep track of visited poses
        
        
        """ Iterate over each pose
                curr_node = new pose
                prev_node = curr_node

            Check if curr_node is unique
                if isnt: curr_node = existing_node
                if is: add to list, add to graph
            
            Check if prev_node is a neigh of curr_node
                if isnt: add bi-edge to each other
                if is: pass
            
            curr_node = new pose
            prev_node = curr_node
        """
        visited_poses = [self.poses[0]]
        prev_node_index,prev_node_pose = 0, self.poses[0]
        curr_node_index = -1
        graph.add_node(prev_node_pose)
        for curr_node_pose in self.poses[1:]:
            print(f"i: {curr_node_index} curr_node: {curr_node_pose} ")
            
            # Check if the node is a duplicate
            is_duplicate = False
            
            for vi,vnode in enumerate(visited_poses):
                if euclidean_distance(curr_node_pose, vnode) < threshold:
                    is_duplicate = True
                    curr_node_pose = vnode
                    curr_node_index = vi
                    break
            

            #  If the node is not a duplicate, add it to the graph and visited poses
            if not is_duplicate:
                curr_node_index = len(visited_poses)
                graph.add_node(curr_node_pose)
                if curr_node_index > 0:
                    graph.add_edge(prev_node_pose, curr_node_pose)
                    graph.add_edge(curr_node_pose, prev_node_pose)
                # graph.add_node(tuple(node))  # Adding the node as a tuple to make it hashable
                visited_poses.append(curr_node_pose)

                print(f"No duplicate on this step, connected node {curr_node_index}: {curr_node_pose} with node {prev_node_index}: {prev_node_pose} \n")
    
            # If the node is a duplicate and not already in the graph, connect it
            elif curr_node_index > 0 and curr_node_index != prev_node_index and not graph.has_edge(curr_node_pose,prev_node_pose):
                print(f"Duplicate on this step, connecting {curr_node_index} and {prev_node_index} \n")
                graph.add_edge(curr_node_pose, prev_node_pose)
                graph.add_edge(prev_node_pose, curr_node_pose)
            else:
                print(f"Duplicate this step, no connection\n")
            prev_node_index = curr_node_index
            prev_node_pose = curr_node_pose
            #print(f"duplicate_index {duplicate_index} duplicate_node {duplicate_vnode} last_added_index {last_added_index} \n")
        
        return graph
    
    def update_redis_data(self):
        # Serialize data if necessary (e.g., JSON)
        if self.graph is None:
            return
        self.redis.set('graph_data', serialize(nx.node_link_data(self.graph)))
        self.redis.set('target', serialize(self.poses[self.target[self.target_index]]))
        self.redis.set('player_position', serialize(self.player_position))
        self.redis.set('player_orientation', serialize(self.direction))

def serialize(data):
    return json.dumps(data)

def euclidean_distance(p1, p2):
    """Calculate the Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


if __name__ == "__main__":
    import vis_nav_game
    import sys
    import os

    # Set an environment variable for macOS to ensure GUI runs in the main thread
    os.environ['PYTHONUNBUFFERED'] = '1'

    # Start plotter.py as a subprocess
    # plotter_process = subprocess.Popen([sys.executable, "plotter.py"])

    # try:
        # Start the main game
    vis_nav_game.play(the_player=KeyboardPlayerPyGame())
    # finally:
        # Ensure that plotter.py is terminated when player.py finishes
        # plotter_process.terminate()
        # plotter_process.wait()

