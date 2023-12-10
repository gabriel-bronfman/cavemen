import redis
import json
import cv2

def deserialize(data):
    return json.loads(data) if data else None

def get_redis_data(redis_conn):
    player_orientation = deserialize(redis_conn.get('player_orientation'))
    return player_orientation

def connect_to_redis():
    return redis.Redis(host='localhost', port=6379, db=0, password='robot_interface')

def main ():
    redis_conn = connect_to_redis()

    upArrow = cv2.imread('arrows/upwards_arrow.png')
    rightArrow = cv2.imread('arrows/right_arrow.png')
    leftArrow = cv2.imread('arrows/left_arrow.png')
    downArrow = cv2.imread('arrows/downward_arrow.png')


    while True:
        
        direction = get_redis_data(redis_conn)

        print(direction)

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