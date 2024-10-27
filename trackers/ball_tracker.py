from ultralytics import YOLO 
import cv2
import pickle
import pandas as pd

class BallTracker:
    def __init__(self,model_path):
        self.model = YOLO(model_path)  # Initialize YOLO model with the given path

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1,[]) for x in ball_positions]  # Extract ball positions from the list of dictionaries
        # convert the list into pandas dataframe
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])  # Convert to DataFrame for easy manipulation

        # interpolate the missing values
        df_ball_positions = df_ball_positions.interpolate()  # Interpolate missing values
        df_ball_positions = df_ball_positions.bfill()  # Fill any remaining NaN values with the next valid observation

        ball_positions = [{1:x} for x in df_ball_positions.to_numpy().tolist()]  # Convert back to original format

        return ball_positions

    def get_ball_shot_frames(self,ball_positions):
        ball_positions = [x.get(1,[]) for x in ball_positions]  # Extract ball positions
        # convert the list into pandas dataframe
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])  # Convert to DataFrame

        df_ball_positions['ball_hit'] = 0  # Initialize ball hit column

        df_ball_positions['mid_y'] = (df_ball_positions['y1'] + df_ball_positions['y2'])/2  # Calculate midpoint of y-coordinates
        df_ball_positions['mid_y_rolling_mean'] = df_ball_positions['mid_y'].rolling(window=5, min_periods=1, center=False).mean()  # Calculate rolling mean
        df_ball_positions['delta_y'] = df_ball_positions['mid_y_rolling_mean'].diff()  # Calculate change in y position

        minimum_change_frames_for_hit = 25  # Define minimum frames for a hit
        for i in range(1,len(df_ball_positions)- int(minimum_change_frames_for_hit*1.2) ):
            negative_position_change = df_ball_positions['delta_y'].iloc[i] >0 and df_ball_positions['delta_y'].iloc[i+1] <0  # Check for negative change
            positive_position_change = df_ball_positions['delta_y'].iloc[i] <0 and df_ball_positions['delta_y'].iloc[i+1] >0  # Check for positive change

            if negative_position_change or positive_position_change:
                change_count = 0 
                for change_frame in range(i+1, i+int(minimum_change_frames_for_hit*1.2)+1):
                    negative_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] >0 and df_ball_positions['delta_y'].iloc[change_frame] <0  # Check for continued negative change
                    positive_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] <0 and df_ball_positions['delta_y'].iloc[change_frame] >0  # Check for continued positive change

                    if negative_position_change and negative_position_change_following_frame:
                        change_count+=1
                    elif positive_position_change and positive_position_change_following_frame:
                        change_count+=1
            
                if change_count>minimum_change_frames_for_hit-1:
                    df_ball_positions['ball_hit'].iloc[i] = 1  # Mark frame as ball hit

        frame_nums_with_ball_hits = df_ball_positions[df_ball_positions['ball_hit']==1].index.tolist()  # Get frame numbers with ball hits

        return frame_nums_with_ball_hits

    def detect_frames(self,frames, read_from_stub=False, stub_path=None):
        ball_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                ball_detections = pickle.load(f)  # Load ball detections from stub file
            return ball_detections

        for frame in frames:
            player_dict = self.detect_frame(frame)  # Detect ball in each frame
            ball_detections.append(player_dict)
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_detections, f)  # Save ball detections to stub file
        
        return ball_detections

    def detect_frame(self,frame):
        results = self.model.predict(frame,conf=0.15)[0]  # Predict using YOLO model with confidence threshold

        ball_dict = {}
        for box in results.boxes:
            result = box.xyxy.tolist()[0]  # Get bounding box coordinates
            ball_dict[1] = result  # Store ball detection result

        return ball_dict

    def draw_bboxes(self,video_frames, player_detections):
        output_video_frames = []
        for frame, ball_dict in zip(video_frames, player_detections):
            # Draw Bounding Boxes
            for track_id, bbox in ball_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Ball ID: {track_id}",(int(bbox[0]),int(bbox[1] -10 )),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)  # Add ball ID text
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)  # Draw bounding box
            output_video_frames.append(frame)
        
        return output_video_frames


    
