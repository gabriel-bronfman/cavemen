import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import json
import redis
import math
import networkx as nx
from matplotlib.backends.backend_agg import FigureCanvasAgg
import heapq


def connect_to_redis():
    return redis.Redis(host='localhost', port=6379, db=0, password="robot_interface")

def deserialize(data):
    return json.loads(data) if data else None

def get_redis_data(redis_conn):
    graph_data = deserialize(redis_conn.get('graph_data'))
    target = deserialize(redis_conn.get('target'))
    player_position = deserialize(redis_conn.get('player_position'))
    player_orientation = deserialize(redis_conn.get('player_orientation'))
    return graph_data, target, player_position, player_orientation

def euclidean_distance(p1, p2):
    """Calculate the Euclidean distance between two points."""
    # if p1 is not None and p2 is not None:
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    #return None

def astar(graph, start, target):

    graph = nx.node_link_graph(graph)

    for node in graph.nodes:
        if start[0] == node[0] and start[1] == node[1]:
            start = node
        elif target[0] == node[0] and target[1] == node[1]:
            target = node
    
    priority_queue = [(0, start)]
    visited = set()
    parents = {start: None}
    g_values = {start: 0}

    while priority_queue:
        current_cost, current_node = heapq.heappop(priority_queue)

        if current_node == target:
            path = []
            while current_node is not None:
                path.append(current_node)
                current_node = parents[current_node]
            return path[::-1]

        if current_node in visited:
            continue

        visited.add(current_node)

        for neighbor in graph.neighbors(current_node):
            new_g = g_values[current_node] + graph[current_node][neighbor].get('weight', 1)

            if neighbor not in g_values or new_g < g_values[neighbor]:
                g_values[neighbor] = new_g
                
                f_value = new_g + euclidean_distance(neighbor, target)
                heapq.heappush(priority_queue, (f_value, neighbor))
                parents[neighbor] = current_node

    return None

def rotate_map(map, direction):
    map = cv2.rotate(map,cv2.ROTATE_90_COUNTERCLOCKWISE)
    if direction == 0:
        pass
    elif direction == 90:
        map = cv2.rotate(map,cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif direction == 180:
        map = cv2.rotate(map,cv2.ROTATE_180)
    else:
        map = cv2.rotate(map,cv2.ROTATE_90_CLOCKWISE)
    return map
        

def draw_graph_with_target(graph_data, target, player_position, player_orientation, window_size=(800, 600), path=None):

    if graph_data is None:
        return None
    # print(graph_data)

    # Create a graph from the graph data
    graph = nx.node_link_graph(graph_data)


    # Extract positions and rotations from nodes
    pos = {node: (node[0], node[1]) for node in graph.nodes()}
    rotations = {node: node[2] for node in graph.nodes()}
    node_colors = []

    for node in graph.nodes:
        if euclidean_distance(node, target) < 25 and euclidean_distance(player_position, node) < 25:
            node_colors.append('green')
        elif euclidean_distance(node, target) < 25:
            node_colors.append('red')
        elif euclidean_distance(player_position, node) < 25:
            node_colors.append('yellow')
        else:
            node_colors.append('blue')
    if path is not None:
        # print("I am doing it")
        for index, node in enumerate(graph.nodes):
            for path_node in path:
                if euclidean_distance(node, path_node) < 25 and euclidean_distance(node, target) > 20:
                    if euclidean_distance(node, player_position) > 25:
                        node_colors[index] = 'black'
                    else:
                        pass


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
    window_size = (800, 600)

    redis_conn = connect_to_redis()
    redis_conn.flushall()

    path_found = False

    while True:

        graph_data, target, player_position, player_orientation = get_redis_data(redis_conn)

        if graph_data and target and player_position:

            if not path_found:
                
                path = astar(graph_data,(0,0),target)
                path_found = True
                print(path)

            img_bgr = draw_graph_with_target(graph_data, target, player_position, player_orientation, window_size, path)
            # img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            #img_bgr = rotate_map(img_bgr, player_orientation)
            img_bgr = cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.imshow('Real-Time Graph', img_bgr)

            if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit on pressing 'q'
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main_plotting_process()