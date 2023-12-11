import math
import redis
from redis.exceptions import AuthenticationError
import json


def deserialize(data):
    return json.loads(data) if data else None

def serialize(data):
    return json.dumps(data)

def connect_to_redis(host='127.0.0.1', port=6379, db=0, password='robot_interface'):
    try:
        connection = redis.Redis(host=host, port=port, db=db)    
    except AuthenticationError: 
        connection = redis.Redis(host=host, port=port, db=db, password=password)
    finally:
        return connection
    
def euclidean_distance(p1, p2):
    """Calculate the Euclidean distance between two points."""
    # if p1 is not None and p2 is not None:
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    #return None