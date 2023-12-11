import json
import math
import os

import redis
from dotenv import load_dotenv
from redis.exceptions import AuthenticationError

# Load environment variables
load_dotenv()


def deserialize(data):
    return json.loads(data) if data else None


def serialize(data):
    return json.dumps(data)


def connect_to_redis():
    host = os.getenv("REDIS_HOST", "127.0.0.1")
    port = int(os.getenv("REDIS_PORT", 6379))
    db = int(os.getenv("REDIS_DB", 0))
    password = os.getenv("REDIS_PASSWORD")  # None if not set

    try:
        # Try to connect without a password if not provided
        connection = redis.Redis(host=host, port=port, db=db, password=password)
        connection.ping()  # Attempt to send a command to check the connection
    except AuthenticationError:
        if password is None:
            raise
        # Try to connect with a password if the first attempt fails
        connection = redis.Redis(host=host, port=port, db=db, password=password)
        connection.ping()  # Check the connection again
    return connection


def euclidean_distance(p1, p2) -> float:
    """Calculate the Euclidean distance between two points."""
    # if p1 is not None and p2 is not None:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    # return None
