import cv2
import os
# from sklearn.cluster import KMeans
from python_orb_slam3 import ORBExtractor
import numpy as np
import faiss

def load_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path, filename))
        
        if img is not None:
            images.append(img)
    return images

def extract_sift_features(images):
    sift = ORBExtractor()
    keypoints_list = []
    descriptors_list = []

    for image in images:
        keypoints, descriptors = sift.detectAndCompute(image, None)
        keypoints_list.append(keypoints)
        descriptors_list.append(descriptors)
        if len(descriptors) < 10 or len(keypoints) < 10:
            print(descriptors)
            print(keypoints)
            cv2.imshow("problem", image)
            cv2.waitKey(0)

    return keypoints_list, descriptors_list

# def create_visual_dictionary(descriptors, num_clusters):
#     kmeans = KMeans(n_clusters=num_clusters, n_init="auto")
#     kmeans.fit(descriptors)
#     return kmeans

def create_visual_dictionary(descriptors, num_clusters):
    d = descriptors.shape[1]  # Dimension of each vector
    kmeans = faiss.Kmeans(d=d, k=num_clusters, niter=300)
    kmeans.train(descriptors.astype(np.float32))
    return kmeans

# def generate_feature_histograms(descriptors, visual_dictionary):
#     num_clusters = visual_dictionary.cluster_centers_.shape[0]
#     histograms = []

#     for desc in descriptors:
#         histogram = np.zeros(num_clusters)
#         labels = visual_dictionary.predict(desc)
#         for label in labels:
#             histogram[label] += 1
#         histograms.append(histogram)

#     return histograms

def generate_feature_histograms(descriptors, visual_dictionary):
    num_clusters = visual_dictionary.k
    histograms = []

    for desc in descriptors:
        histogram = np.zeros(num_clusters)
        _, labels = visual_dictionary.index.search(desc.astype(np.float32), 1)
        for label in labels.flatten():
            histogram[label] += 1
        histograms.append(histogram)

    return histograms

def compare_histograms(query_histogram, list_of_histograms):
    # Calculate Euclidean distances
    distances = [np.linalg.norm(query_histogram - hist) for hist in list_of_histograms]
    
    # Find the index of the most similar histogram
    most_similar_index = np.argmin(distances)
    
    return most_similar_index

# def process_image_and_find_best_match(new_image, list_of_histograms, kmeans):
#     # Step 1: Extract features from the new image
#     sift = ORBExtractor()
#     keypoints, descriptors = sift.detectAndCompute(new_image, None)
    
#     # Step 2: Generate the feature histogram for the new image
#     keypoints, descriptors = sift.detectAndCompute(new_image, None)
    
#     # Step 2: Generate the feature histogram for the new image
#     num_clusters = kmeans.cluster_centers_.shape[0]
#     histogram = np.zeros(num_clusters)
#     labels = kmeans.predict(descriptors)
#     for label in labels:
#         histogram[label] += 1
    
#     # Step 3: Compare the histogram to the list of histograms
#     distances = [np.linalg.norm(histogram - hist) for hist in list_of_histograms]
    
#     # Find the indices of the 5 best candidates
#     best_candidates_indices = np.argsort(distances)[:3]
    
#     return np.array(best_candidates_indices)

def process_image_and_find_best_match(new_image, list_of_histograms, kmeans):
    # Step 1: Extract features from the new image
    sift = ORBExtractor()
    keypoints, descriptors = sift.detectAndCompute(new_image, None)
    
    # Ensure descriptors are in the correct format (np.float32)
    descriptors = descriptors.astype(np.float32)

    # Step 2: Generate the feature histogram for the new image
    num_clusters = kmeans.k
    histogram = np.zeros(num_clusters)

    # Use FAISS to find nearest clusters
    _, labels = kmeans.index.search(descriptors, 1)
    for label in labels.flatten():
        histogram[label] += 1
    
    # Step 3: Compare the histogram to the list of histograms
    distances = [np.linalg.norm(histogram - hist) for hist in list_of_histograms]
    
    # Find the indices of the 3 best candidates
    best_candidates_indices = np.argsort(distances)[:3]
    
    return np.array(best_candidates_indices)

    
    
# For testing only
def main():
    folder_path = 'data/textures'
    images = load_images_from_folder(folder_path)
    keypoints, descriptors = extract_sift_features(images)
    visual_dictionary = create_visual_dictionary(np.vstack(descriptors), num_clusters=100)
    histograms = generate_feature_histograms(descriptors, visual_dictionary)
    
    best_indexes = process_image_and_find_best_match(images[155], histograms, visual_dictionary)

    for index in best_indexes:
        cv2.imshow(f"{index}th best guess", images[index])
        cv2.waitKey(0)


if __name__ == "__main__":
    main()

