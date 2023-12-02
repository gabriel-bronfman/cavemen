import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import redis
import math
import networkx as nx
from matplotlib.backends.backend_agg import FigureCanvasAgg
import pygame

def connect_to_redis():
    return redis.Redis(host='localhost', port=6379, db=0)

def deserialize(data):
    return json.loads(data) if data else None

def get_redis_data(redis_conn):
    graph_data = deserialize(redis_conn.get('graph_data'))
    target = deserialize(redis_conn.get('target'))
    player_position = deserialize(redis_conn.get('player_position'))
    return graph_data, target, player_position

def euclidean_distance(p1, p2):
    """Calculate the Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def draw_graph_with_target(graph_data, target, player_position, window_size=(800, 600)):
    if graph_data is None:
        return None
    print(graph_data)

    # Create a graph from the graph data
    graph = nx.node_link_graph(graph_data)


    # Extract positions and rotations from nodes
    pos = {node: (node[0], node[1]) for node in graph.nodes()}
    rotations = {node: node[2] for node in graph.nodes()}
    node_colors = []

    for node in graph.nodes:
        if euclidean_distance(node, target) < 25:
            node_colors.append('red')
        elif euclidean_distance(player_position, node) < 25:
            node_colors.append('yellow')
        else:
            node_colors.append('blue')

    # Draw the graph
    nx.draw(graph, pos, node_color=node_colors, node_size=400, arrowstyle='<|-|>', arrowsize=15)

    # Get the Matplotlib figure and axis
    fig = plt.gcf()
    ax = plt.gca()

    # Convert the Matplotlib figure to a NumPy array
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    img_width, img_height = fig.get_size_inches() * fig.get_dpi()
    img_array = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(int(img_height), int(img_width), 4)

    # Convert RGBA to BGR (OpenCV uses BGR order)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
    return img_bgr


def main_plotting_process():
    pygame.init()
    window_size = (800, 600)
    # screen = pygame.display.set_mode(window_size)
    # pygame.display.set_caption('Mapping')

    redis_conn = connect_to_redis()

    while True:
        # Check for Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        graph_data, target, player_position = get_redis_data(redis_conn)

        if graph_data and target and player_position:
            img_bgr = draw_graph_with_target(graph_data, target, player_position, window_size)
            # img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            cv2.imshow('Real-Time Graph', img_bgr)

            if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit on pressing 'q'
                break

    cv2.destroyAllWindows()

        # pygame.time.wait(10)  # Wait a bit to not consume too much CPU


if __name__ == "__main__":
    main_plotting_process()