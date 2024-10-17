from ultralytics import YOLO  # Importing the YOLO model from the Ultralytics library for object detection and tracking.
import cv2  # Importing OpenCV library for video processing and image manipulation.
import pickle  # Importing the pickle module for saving and loading data (used to store player detection data).
import sys  # Importing sys for system-related functionalities, such as manipulating the Python path.
sys.path.append('../')  # Adding the parent directory to the system path to access modules (like 'utils') from there.
from utils import measure_distance, get_center_of_bbox  # Importing helper functions to calculate distances and get the center of bounding boxes.

# Define a class to track players in tennis videos using the YOLO model
class PlayerTracker:
    def __init__(self, model_path):
        """
        Constructor to initialize the PlayerTracker object with a YOLO model.
        :param model_path: Path to the pre-trained YOLO model.
        """
        self.model = YOLO(model_path)  # Load the YOLO model from the provided model path for player detection.

    def choose_and_filter_players(self, court_keypoints, player_detections):
        """
        Filters player detections to keep only the selected players based on their proximity to the court keypoints.
        :param court_keypoints: List of court coordinates (keypoints) for measuring proximity to players.
        :param player_detections: List of dictionaries containing bounding boxes for each detected player across frames.
        :return: A filtered list of player detections.
        """
        player_detections_first_frame = player_detections[0]  # Get the player detections from the first frame for selection.
        chosen_player = self.choose_players(court_keypoints, player_detections_first_frame)  # Select the two closest players to the court keypoints.
        filtered_player_detections = []  # Initialize a list to store filtered player detections.

        # Iterate over each frame's player detections
        for player_dict in player_detections:
            # Filter the player detections, keeping only the chosen players
            filtered_player_dict = {track_id: bbox for track_id, bbox in player_dict.items() if track_id in chosen_player}
            filtered_player_detections.append(filtered_player_dict)  # Append the filtered detections to the list.

        return filtered_player_detections  # Return the list of filtered player detections.

    def choose_players(self, court_keypoints, player_dict):
        """
        Choose the two players that are closest to the court keypoints.
        :param court_keypoints: List of court keypoints used to measure proximity.
        :param player_dict: Dictionary containing player bounding boxes and their track IDs.
        :return: A list of chosen player track IDs.
        """
        distances = []  # Initialize a list to store distances between players and court keypoints.

        # Loop through each player detected in the current frame
        for track_id, bbox in player_dict.items():
            player_center = get_center_of_bbox(bbox)  # Get the center coordinates of the player's bounding box.

            min_distance = float('inf')  # Initialize the minimum distance to infinity for comparison.
            # Loop through the court keypoints, calculating the distance from the player to the court
            for i in range(0, len(court_keypoints), 2):
                court_keypoint = (court_keypoints[i], court_keypoints[i+1])  # Get each court keypoint (x, y).
                distance = measure_distance(player_center, court_keypoint)  # Measure the distance from the player to the keypoint.
                if distance < min_distance:  # If this distance is smaller than the current minimum distance, update it.
                    min_distance = distance
            distances.append((track_id, min_distance))  # Store the track ID and corresponding minimum distance.

        # Sort the distances list in ascending order based on distance to find the closest players.
        distances.sort(key=lambda x: x[1])
        # Select the two players with the smallest distances (i.e., the closest to the court keypoints).
        chosen_players = [distances[0][0], distances[1][0]]
        return chosen_players  # Return the track IDs of the chosen players.

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        """
        Detect players in the given frames, or load the detections from a saved file (stub) if available.
        :param frames: List of video frames to process.
        :param read_from_stub: Boolean indicating whether to read detections from a pre-saved file.
        :param stub_path: Path to the stub file to read from or save to.
        :return: A list of player detections for each frame.
        """
        player_detections = []  # Initialize an empty list to store player detections.

        # If specified, load player detections from the saved file (stub)
        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)  # Load the player detections from the file using pickle.
            return player_detections  # Return the loaded player detections.

        # If no stub is used, detect players in each frame
        for frame in frames:
            player_dict = self.detect_frame(frame)  # Detect players in the current frame.
            player_detections.append(player_dict)  # Append the player detections to the list.

        # If a stub path is provided, save the detections to a file for future use
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)  # Save the player detections using pickle.

        return player_detections  # Return the list of player detections.

    def detect_frame(self, frame):
        """
        Detect players in a single video frame using the YOLO model.
        :param frame: A single video frame (image) to process.
        :return: A dictionary of player track IDs and their bounding boxes.
        """
        results = self.model.track(frame, persist=True)[0]  # Use the YOLO model to track objects in the frame.
        id_name_dict = results.names  # Get the mapping of object class IDs to names.

        player_dict = {}  # Initialize an empty dictionary to store player detections.
        # Loop through all detected bounding boxes
        for box in results.boxes:
            track_id = int(box.id.tolist()[0])  # Get the track ID for the detected object (as integer).
            result = box.xyxy.tolist()[0]  # Get the bounding box coordinates (x1, y1, x2, y2).
            object_cls_id = box.cls.tolist()[0]  # Get the object class ID.
            object_cls_name = id_name_dict[object_cls_id]  # Map the class ID to the class name.
            if object_cls_name == "person":  # If the detected object is a person, store the track ID and bounding box.
                player_dict[track_id] = result

        return player_dict  # Return the dictionary of player track IDs and bounding boxes.

    def draw_bboxes(self, video_frames, player_detections):
        """
        Draw bounding boxes around players in the given video frames.
        :param video_frames: List of video frames where bounding boxes will be drawn.
        :param player_detections: List of player detections for each frame.
        :return: A list of video frames with bounding boxes drawn on them.
        """
        output_video_frames = []  # Initialize a list to store the output frames with bounding boxes.

        # Iterate over each frame and its corresponding player detections
        for frame, player_dict in zip(video_frames, player_detections):
            # Loop through each player and draw their bounding box on the frame
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox  # Extract the coordinates of the bounding box.
                # Put text with the player track ID near the bounding box
                cv2.putText(frame, f"Player ID: {track_id}", (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                # Draw the bounding box rectangle around the detected player
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            output_video_frames.append(frame)  # Append the processed frame to the list.

        return output_video_frames  # Return the list of video frames with bounding boxes.