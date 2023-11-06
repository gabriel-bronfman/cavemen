from vis_nav_game import Player, Action
import pygame
import cv2
import numpy as np
from place_recognition import extract_sift_features, create_visual_dictionary, generate_feature_histograms, compare_histograms, process_image_and_find_best_match


class KeyboardPlayerPyGame(Player):
    def __init__(self):
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None
        self.keymap = None
        self.images = []
        self.visual_dictionary = None
        self.histograms = None
        super(KeyboardPlayerPyGame, self).__init__()

    def reset(self):
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None

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
                    self.last_act |= self.keymap[event.key]
                else:
                    self.show_target_images()
            if event.type == pygame.KEYUP:
                if event.key in self.keymap:
                    self.last_act ^= self.keymap[event.key]
        return self.last_act

    def show_target_images(self):
        
        targets = self.get_target_images()
        best_indexes = [[]]
        for target in targets:
            best_index = process_image_and_find_best_match(target,self.histograms,self.visual_dictionary)

            best_indexes.append(best_index)

        if targets is None or len(targets) <= 0:
            return
        
        hor1 = cv2.vconcat([self.images[best_indexes[1][0]],self.images[best_indexes[2][0]]])
        hor2 = cv2.vconcat([self.images[best_indexes[3][0]],self.images[best_indexes[4][0]]])
        concat_img = cv2.hconcat([hor1, hor2])

        hor1_target = cv2.hconcat(targets[:2])
        hor2_target = cv2.hconcat(targets[2:])
        concat_img_target = cv2.vconcat([hor1_target, hor2_target])

        w, h = concat_img_target.shape[:2]
        
        color = (0, 0, 0)

        concat_img_target = cv2.line(concat_img_target, (int(h/2), 0), (int(h/2), w), color, 2)
        concat_img_target = cv2.line(concat_img_target, (0, int(w/2)), (h, int(w/2)), color, 2)

        w_offset = 25
        h_offset = 10
        font = cv2.FONT_HERSHEY_SIMPLEX
        line = cv2.LINE_AA
        size = 0.75
        stroke = 1

        cv2.putText(concat_img_target, 'Front View', (h_offset, w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img_target, 'Right View', (int(h/2) + h_offset, w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img_target, 'Back View', (h_offset, int(w/2) + w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img_target, 'Left View', (int(h/2) + h_offset, int(w/2) + w_offset), font, size, color, stroke, line)

        cv2.imshow(f'KeyboardPlayer:target_images', concat_img_target)
        cv2.imshow(f'KeyboardPlayer:Recognized Location', concat_img)
        cv2.waitKey(1)

    def set_target_images(self, images):
        super(KeyboardPlayerPyGame, self).set_target_images(images)
        self.pre_navigation_fuck_you()
        self.find_targets()
        self.show_target_images()

    def pre_navigation_fuck_you(self) -> None:
        if len(self.images) != 0:
            print(f"Finding descriptors for {len(self.images)} images")
            keypoints,descriptors = extract_sift_features(self.images)
            print(f"Creating dictionary for images")
            self.visual_dictionary = create_visual_dictionary(np.vstack(descriptors), num_clusters=100)
            print(f"Creating {len(self.images)} histograms")
            self.histograms = generate_feature_histograms(keypoints, descriptors, self.visual_dictionary)
            
        

    def find_targets(self):
        targets = self.get_target_images()
        for target in targets:
            best_indexes = process_image_and_find_best_match(target,self.histograms,self.visual_dictionary)
            cv2.imshow("best target", self.images[best_indexes[0]])



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
        rgb = convert_opencv_img_to_pygame(fpv)
        self.screen.blit(rgb, (0, 0))
        pygame.display.update()


if __name__ == "__main__":
    import vis_nav_game
    vis_nav_game.play(the_player=KeyboardPlayerPyGame())
