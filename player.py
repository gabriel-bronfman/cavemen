from vis_nav_game import Player, Action
import pygame
import cv2
import redis
import struct
import numpy as np
import plotly.graph_objects as go

ROTATE_VALUE = 1.5
MOVE_VALUE = 10

class KeyboardPlayerPyGame(Player):
    def __init__(self):
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None
        self.keymap = None
        # self.redis_stream = redis.Redis(host='localhost', port=6379, db=0)

        # Initialize the map data
        self.map_size = (1000, 1000, 3)  # Example size for a larger map
        self.map_data = np.zeros(self.map_size, dtype=np.uint8)
        self.direction = 0  # Represents the current angle in degrees
        self.map_scale = 4  # Each unit in the map_data will be a 4x4 pixel square in the OpenCV window
        self.player_position = (self.map_size[0] // 2, self.map_size[1] // 2) # Start in the middle of the map
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
        self.player_position = (self.map_size[0] // 2, self.map_size[1] // 2)
        self.draw_map(color=[0, 0, 255])

        pygame.init()

        self.keymap = {
            pygame.K_LEFT: Action.LEFT,
            pygame.K_RIGHT: Action.RIGHT,
            pygame.K_UP: Action.FORWARD,
            pygame.K_DOWN: Action.BACKWARD,
            pygame.K_SPACE: Action.CHECKIN,
            pygame.K_ESCAPE: Action.QUIT
        }

    def act(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self.last_act = Action.QUIT
                return Action.QUIT

            if event.type == pygame.KEYDOWN:
                if event.key in self.keymap:
                    self.key_hold_state[event.key] = True
                    self.key_hold_time[event.key]['start'] = pygame.time.get_ticks()
                    self.last_act |= self.keymap[event.key]
                else:
                    self.show_target_images()
                    
            if event.type == pygame.KEYUP:
                if event.key in self.keymap:
                    self.key_hold_state[event.key] = False
                    self.key_hold_time[event.key]['end'] = pygame.time.get_ticks()
                    self.last_act ^= self.keymap[event.key]
            self.update_map_on_keypress()
        
        return self.last_act

    def show_target_images(self):
        self.player_position = (self.map_size[0] // 2, self.map_size[1] // 2)
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
        cv2.putText(concat_img, 'Right View', (int(h/2) + h_offset, w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Back View', (h_offset, int(w/2) + w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Left View', (int(h/2) + h_offset, int(w/2) + w_offset), font, size, color, stroke, line)

        cv2.imshow(f'KeyboardPlayer:target_images', concat_img)
        cv2.waitKey(1)

    def set_target_images(self, images):
        super(KeyboardPlayerPyGame, self).set_target_images(images)
        self.show_target_images()

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
        # print(np.shape(fpv))
        # self.toRedis(self.redis_stream, fpv , 'curr_frame')
        rgb = convert_opencv_img_to_pygame(fpv)
        self.screen.blit(rgb, (0, 0))
        pygame.display.update()

    def toRedis(self,r, a, n):
        # Store given Numpy array 'a' in Redis under key 'n'
        h, w = a.shape[:2]
        shape = struct.pack('>II',h,w)
        encoded = shape + a.tobytes()

        # Store encoded data in Redis
        r.set(n,encoded)
        return
    
    def update_map_on_keypress(self):
        # Rotate left or right based on the current key hold state
        self.direction = (ROTATE_VALUE*(self.key_hold_time[pygame.K_RIGHT]['end'] - self.key_hold_time[pygame.K_RIGHT]['start']) \
            - ROTATE_VALUE*(self.key_hold_time[pygame.K_LEFT]['end'] - self.key_hold_time[pygame.K_LEFT]['start']))
        # if self.key_hold_state[pygame.K_LEFT]:
        #     self.direction -= ROTATE_VALUE  # Rotate 1 degree left
        # if self.key_hold_state[pygame.K_RIGHT]:
        #     self.direction += ROTATE_VALUE  # Rotate 1 degree right

        # Ensure the direction is within 0-359 degrees
        self.direction %= 360

        # Move forward or backward based on the current direction
        move_x, move_y = 0, 0
        if self.key_hold_state[pygame.K_UP] or self.key_hold_state[pygame.K_DOWN]:
            move_amount = MOVE_VALUE if self.key_hold_state[pygame.K_UP] else -MOVE_VALUE
            move_x = int(np.round(move_amount * np.cos(np.deg2rad(self.direction))))
            move_y = int(np.round(move_amount * np.sin(np.deg2rad(self.direction))))

        # Update the player's position on the map
        self.player_position = (
            max(0, min(self.player_position[0] + move_x, self.map_size[0] - 1)),
            max(0, min(self.player_position[1] + move_y, self.map_size[1] - 1))
        )

        self.draw_map()

    def draw_map(self, color=[255, 255, 255]):
        # Determine the size of the window we are using to display the map
        window_size = self.map_size[:2]  # Size of the map window in pixels
        if sum(self.map_data[self.player_position[1], self.player_position[0]]) == 0:
            self.map_data[self.player_position[1], self.player_position[0]] = color
        # Determine the bounds for the centering effect based on the scale and window size
        center_bounds = (self.map_size[0] - window_size[0] // self.map_scale,
                        self.map_size[1] - window_size[1] // self.map_scale)

        # Get player's position on the map_data
        player_x, player_y = self.player_position

        # Calculate the top-left coordinate of the map view
        top_left_x = max(0, min(player_x - window_size[0] // (2 * self.map_scale), center_bounds[0]))
        top_left_y = max(0, min(player_y - window_size[1] // (2 * self.map_scale), center_bounds[1]))

        # Create a view from the map data that corresponds to the current window view
        view = self.map_data[top_left_y:top_left_y + window_size[1] // self.map_scale,
                            top_left_x:top_left_x + window_size[0] // self.map_scale]

        # Create an image that enlarges the map using the scale factor
        map_image_large = cv2.resize(view, window_size, interpolation=cv2.INTER_NEAREST)

        # Determine the position to draw the player in the view
        # draw_x = min(max(player_x - top_left_x, 0), window_size[0] // self.map_scale) * self.map_scale
        # draw_y = min(max(player_y - top_left_y, 0), window_size[1] // self.map_scale) * self.map_scale

        # Draw the player on the map view
        # map_image_large[draw_y:draw_y+self.map_scale, draw_x:draw_x+self.map_scale] = color  # Mark the new position

        # Display the map image using OpenCV
        cv2.imshow('2D Map', map_image_large)
        cv2.waitKey(1)  # Refresh the OpenCV window

if __name__ == "__main__":
    import vis_nav_game
    vis_nav_game.play(the_player=KeyboardPlayerPyGame())
