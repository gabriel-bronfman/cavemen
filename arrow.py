import redis
import json
import cv2
from typing import Any

from utils import deserialize, connect_to_redis

def get_redis_data(redis_conn: redis.Redis) -> int:
    player_orientation = deserialize(redis_conn.get('player_orientation'))
    return player_orientation


def main():
    redis_conn = connect_to_redis()

    upArrow = cv2.imread('assets/img/arrows/upwards_arrow.png')
    rightArrow = cv2.imread('assets/img/arrows/right_arrow.png')
    leftArrow = cv2.imread('assets/img/arrows/left_arrow.png')
    downArrow = cv2.imread('assets/img/arrows/downward_arrow.png')

    while True:
        
        direction = get_redis_data(redis_conn)

        if direction == 0:
            cv2.imshow('direction', upArrow)
        elif direction == 270:
            cv2.imshow('direction',rightArrow)
        elif direction == 180:
            cv2.imshow('direction',downArrow)
        elif direction == 90:
            cv2.imshow('direction', leftArrow)
        else:
            continue
        cv2.moveWindow('direction',0,600)
        cv2.waitKey(1)

if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()