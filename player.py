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

        # Initialize the map data
        self.map_size = (1000, 1000, 3)  # Example size for a larger map
        self.map_data = np.zeros(self.map_size, dtype=np.uint8)
        self.direction = 0  # Represents the current angle in degrees
        self.map_scale = 4  # Each unit in the map_data will be a 4x4 pixel square in the OpenCV window

        
        self.player_position = (0,0)
        self.key_hold_state = {pygame.K_LEFT: False, pygame.K_RIGHT: False, pygame.K_UP: False, pygame.K_DOWN: False}
        self.key_hold_time = {pygame.K_LEFT: {'start':0, 'end':0}, pygame.K_RIGHT: {'start':0, 'end':0}, pygame.K_UP: {'start':0, 'end':0}, pygame.K_DOWN: {'start':0, 'end':0}}
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

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self.last_act = Action.QUIT
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
                            self.turn_count = 1
                            self.direction += 90
                        elif self.keymap[event.key] == Action.RIGHT:
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
    

    def show_target_images(self):
        self.player_position = (0,0)
        self.direction = 0
        targets = self.get_target_images()
        if self.self_validate and len(targets) > 0:
            targets[0] = self.validation_img

        best_indexes = [[]]
        
        for target in targets:

            if self.visual_dictionary is None:
                raise AttributeError("Dictionary was None")
            
            best_index = process_image_and_find_best_match(target,self.histograms,self.visual_dictionary)
            best_indexes.append(best_index)

        if targets is None or len(targets) == 0:
            return

        # Concatenate best match images in pairs horizontally and then vertically
        hor1 = cv2.hconcat([self.images[best_indexes[1][0]], self.images[best_indexes[2][0]]])
        hor2 = cv2.hconcat([self.images[best_indexes[3][0]], self.images[best_indexes[4][0]]])
        concat_img = cv2.vconcat([hor1, hor2])

        # Concatenate second best match images similarly as above
        hor1_second_best = cv2.hconcat([self.images[best_indexes[1][1]], self.images[best_indexes[2][1]]])
        hor2_second_best = cv2.hconcat([self.images[best_indexes[3][1]], self.images[best_indexes[4][1]]])
        concat_img_second_best = cv2.vconcat([hor1_second_best, hor2_second_best])

        # Concatenate target images similarly as above
        hor1_target = cv2.hconcat(targets[:2])
        hor2_target = cv2.hconcat(targets[2:])
        concat_img_target = cv2.vconcat([hor1_target, hor2_target])

        # Concatenate third best match images similarly as above
        hor1_third_best = cv2.hconcat([self.images[best_indexes[1][2]], self.images[best_indexes[2][2]]])
        hor2_third_best = cv2.hconcat([self.images[best_indexes[3][2]], self.images[best_indexes[4][2]]])
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
                            f'X: {self.poses[best_indexes[index][rank]][0]:.2f} Y: {self.poses[best_indexes[index][rank]][1]:.2f} W: {self.poses[best_indexes[index][rank]][2]:.2f}\t', 
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

    def set_target_images(self, images):
        super(KeyboardPlayerPyGame, self).set_target_images(images)
        # self.pre_navigation_bypass()
        # self.find_targets()
        # self.draw_map()
        graph = self.create_graph_from_poses()
        draw_graph_with_rotations(graph)

        self.show_target_images()

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
            self.histograms = generate_feature_histograms(keypoints, descriptors, self.visual_dictionary)   

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
        pygame.display.update()

    def update_map_on_keypress(self):
        # Rotate left or right based on the current key hold state
        # self.direction = (ROTATE_VALUE*(self.key_hold_time[pygame.K_RIGHT]['end'] - self.key_hold_time[pygame.K_RIGHT]['start']) \
        #     - ROTATE_VALUE*(self.key_hold_time[pygame.K_LEFT]['end'] - self.key_hold_time[pygame.K_LEFT]['start']))
        # if self.get_state():
        #     fps = self.get_state()[4]
        # else:
        #     fps = 35
        # spf = 1/fps

        # if self.key_hold_state[pygame.K_LEFT]:
        #     self.direction += ROTATE_VALUE  # Rotate 1 degree left 
        # if self.key_hold_state[pygame.K_RIGHT]:
        #     self.direction -= ROTATE_VALUE  # Rotate 1 degree right

        # Ensure the direction is within 0-359 degrees
        self.direction %= 360

        # Move forward or backward based on the current direction
        move_x, move_y = 0, 0
        if self.key_hold_state[pygame.K_UP] or self.key_hold_state[pygame.K_DOWN]:
            move_amount = MOVE_VALUE if self.key_hold_state[pygame.K_UP] else -MOVE_VALUE
            move_x = move_amount * np.cos(np.deg2rad(self.direction))
            move_y = move_amount * np.sin(np.deg2rad(self.direction))
        # print(f'move_x: {move_x} move_y: {move_y}')
        self.player_position = (self.player_position[0] + move_x, self.player_position[1] + move_y)
        # # Update the player's position on the map
        # self.player_position = (
        #     max(0, min(self.player_position[0] + move_x, self.map_size[0] - 1)),
        #     max(0, min(self.player_position[1] + move_y, self.map_size[1] - 1))
        # )

        sys.stdout.write(f'\rX: {self.player_position[0]:.2f} Y:{self.player_position[1]:.2f} W: {self.direction:.2f}')
        sys.stdout.flush()

    # def draw_map(self):
    #     if self.poses is None or len(self.poses) == 0:
    #         return
    #     # Threshold distance for combining nearby nodes
    #     threshold_distance = 10  # Adjust this value as needed

    #     # Create a directed graph
    #     G = nx.DiGraph()

    #     # Instantiate lists
    #     combined_poses = [(self.poses[0][0],self.poses[0][1],self.poses[0][2])]
    #     G.add_node(0, pos=(self.poses[0][0],self.poses[0][1]), rot=self.poses[0][2])
        
    #     visited_list = []

    #     for index in range(len(self.poses[1:])):
    #         close_to_existing = False
    #         duplicate_index = -1
    #         curr_x, curr_y, curr_rot = self.poses[index]
    #         print(index)

    #         for i, (cx, cy, crot) in enumerate(combined_poses):
    #             if math.sqrt((cx - curr_x)**2 + (cy - curr_y)**2) < threshold_distance:
    #                 close_to_existing = True
    #                 duplicate_index = i
    #                 break
            
    #         if not close_to_existing:
    #             visited_list.append(index)
    #             combined_poses.append((curr_x, curr_y, curr_rot))
    #             G.add_node(index,pos=(curr_x,curr_y), rot=curr_rot)
    #             G.add_edge(index-1,index)
    #             G.add_edge(index,index-1)

    #         elif duplicate_index > -1 and not G.has_edge(duplicate_index,index-1):
    #             visited_list.append(duplicate_index)
    #             G.add_edge(index-1,duplicate_index)
    #             G.add_edge(duplicate_index,index-1)
    #             duplicate_index = -1

    #     print(G)

    #     # Draw the graph
    #     pos = nx.get_node_attributes(G, 'pos')
    #     rotations = nx.get_node_attributes(G, 'rotation')
    #     nx.draw(G, pos, node_color='skyblue', node_size=700, arrowstyle='<|-|>', arrowsize=15)

    #     # Add rotation indication to each node
    #     for n, (x, y) in pos.items():
    #         rotation = rotations[n]
    #         dx = math.cos(math.radians(rotation)) * 0.1  # Length of the rotation indicator
    #         dy = math.sin(math.radians(rotation)) * 0.1
    #         plt.arrow(x, y, dx, dy, head_width=0.05, head_length=0.1, fc='black', ec='black')

    #     plt.show()


    def create_graph_from_poses(self, threshold=10):
        if self.poses is None or len(self.poses) == 0:
            return None
        """
        Create a graph from a list of poses where each pose is [x, y, rotation].
        Connect nodes if their Euclidean distance is less than a given threshold.
        """
        # Create an empty graph
        graph = nx.Graph()

        # List to keep track of visited poses
        visited_poses = []

        # Iterate over each pose
        for i, node in enumerate(self.poses):
            # Check if the node is a duplicate
            is_duplicate = False
            duplicate_vnode = None

            for vnode in visited_poses:
                if euclidean_distance(node, vnode) < threshold:
                    is_duplicate = True
                    duplicate_vnode = vnode
                    break

            # If the node is not a duplicate, add it to the graph and visited poses
            if not is_duplicate:
                graph.add_node(tuple(node))
                if i > 0:
                    graph.add_edge(self.poses[i-1], node)
                    graph.add_edge(node, self.poses[i-1])
                # graph.add_node(tuple(node))  # Adding the node as a tuple to make it hashable
                visited_poses.append(node)

            # If the node is a duplicate and not already in the graph, connect it
            elif i > 0 and duplicate_vnode and duplicate_vnode != node and not graph.has_edge(duplicate_vnode,node):
                graph.add_edge(node, duplicate_vnode)
                graph.add_edge(duplicate_vnode, node)
        return graph

def draw_graph_with_rotations(graph):
    if graph is None:
        return
    """
    Draw the graph with an indication of rotation at each node.
    """
    # Extract positions and rotations from nodes
    pos = {node: (node[0], node[1]) for node in graph.nodes()}
    rotations = {node: node[2] for node in graph.nodes()}

    # Draw the graph
    nx.draw(graph, pos, node_color='skyblue', node_size=700, arrowstyle='<|-|>', arrowsize=15)

    # Add rotation indication to each node
    for node, (x, y) in pos.items():
        rotation = rotations[node]
        dx = math.cos(math.radians(rotation)) * 0.1  # Length of the rotation indicator
        dy = math.sin(math.radians(rotation)) * 0.1
        plt.arrow(x, y, dx, dy, head_width=0.01, head_length=0.1, fc='black', ec='black')

    plt.show()

def euclidean_distance(p1, p2):
    """Calculate the Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


if __name__ == "__main__":
    import vis_nav_game
    vis_nav_game.play(the_player=KeyboardPlayerPyGame())
