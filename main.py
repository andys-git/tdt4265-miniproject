from ultralytics import YOLO

from make_movie_general import images_to_video
from object_detection_pipeline import detect_field_objects
from orientation_detection_pipeline import load_frames, detect_field_points, estimate_homographies, load_frame_dict, save_frame_dict
from generate_field import get_field_markers
from generate_overlay import unwarp_all, plot_frames_on_field
from make_movie_yolo import make_video_yolo

image_folder_path = r"C:\Users\andys\Documents\TDT4265\clean_data\hamkam"
object_model = "yolo_model_full_1_cpt5.pt"
point_model = "yolo_model_points_2.pt"

CONF_THRESH = {
    "player": 0.25,
    "referee": 0.25,
    "ball": 0.20
}

FIELD_DIMENSIONS = {'x': 105.0, 'y': 68.0}

def main():
    frames = load_frames(image_folder_path)

    try:
        frame_dict = load_frame_dict()
    except FileNotFoundError:
        # Initialize our frames with detected keypoints
        print("Detecting keypoints...")
        frame_dict = detect_field_points(YOLO(point_model), frames, image_folder_path)

        # Add attribute to each frame how it has been warped (homographic matrix)
        print("Estimating field orientation...")
        warp = estimate_homographies(frame_dict, frames, get_field_markers(FIELD_DIMENSIONS), image_folder_path)
        for f, H in zip(frames, warp): frame_dict[f]['warp'] = H

        objects_dict = detect_field_objects(YOLO(object_model), frames, image_folder_path, CONF_THRESH)
        for f in frames:
            frame_dict[f]['objects'] = objects_dict.get(f, [])

        frame_dict = unwarp_all(frame_dict)
        save_frame_dict(frame_dict)

    # print("Making YOLO video frames...")
    # make_video_yolo(
    #     image_folder_path,
    #     object_model,
    #     0,
    #     1801,
    #     classes_to_draw=['player', 'referee', 'ball'],
    #     image_output_dir=r"C:\Users\andys\Documents\TDT4265\final_result_yolo"
    # )
    #
    # print("Making plot frames...")
    # plot_frames_on_field(
    #     frame_dict,
    #     r"C:\Users\andys\Documents\TDT4265\final_result_yolo",
    #     FIELD_DIMENSIONS,
    #     background_path='field.jpg',
    #     output_dir=r"C:\Users\andys\Documents\TDT4265\final_result_overlay"
    # )

    print("Making final video...")
    images_to_video(
        r"C:\Users\andys\Documents\TDT4265\final_result_overlay",
        "final_result_short.mp4",
        fps=24
    )

    return

if __name__ == "__main__":
    main()