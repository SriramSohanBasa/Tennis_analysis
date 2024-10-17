from utils import (read_video, 
                   save_video
                #    measure_distance,
                #    draw_player_stats,
                #    convert_pixel_distance_to_meters
                   )


def main():

    # Read Video
    input_video_path = "input_videos/input_video.mp4"
    video_frames = read_video(input_video_path)
    
    save_video(video_frames, "output_videos/output_video.avi")

if __name__ == "__main__":
    main()