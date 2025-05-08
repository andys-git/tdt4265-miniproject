import os
import numpy as np
import cv2
from old_scripts.penalty_arc_calculate import circle_rectangle_intersections
import math


def invert_y(y, field_dim):
    return round(field_dim['y'] - y, 2)

def generate_field_image(field_dimensions, image_width, image_height, line_thickness=2):
    # Points
    bottom_left = (0.0, invert_y(0, field_dimensions))
    bottom_right = (field_dimensions['x'], invert_y(0, field_dimensions))
    top_left = (0.0, invert_y(field_dimensions['y'], field_dimensions))
    top_right = (field_dimensions['x'], invert_y(field_dimensions['y'], field_dimensions))

    top_middle = (field_dimensions['x'] / 2, invert_y(field_dimensions['y'], field_dimensions))
    bottom_middle = (field_dimensions['x'] / 2, invert_y(0, field_dimensions))
    middle = (field_dimensions['x'] / 2, invert_y(field_dimensions['y'] / 2, field_dimensions))
    circle_radius = 9.15

    penalty_box_height = (2 * 11) + (2 * 5.5) + 7.32
    penalty_box_width = 5.5 + 11
    penalty_spot_distance_to_goal_line = 11

    left_penalty_middle = (0, invert_y(field_dimensions['y'] / 2, field_dimensions))
    left_penalty_bottom_left = (0, invert_y(left_penalty_middle[1] + penalty_box_height / 2, field_dimensions))
    left_penalty_bottom_right = (penalty_box_width, invert_y(left_penalty_middle[1] + penalty_box_height / 2, field_dimensions))
    left_penalty_top_left = (0, invert_y(left_penalty_middle[1] - penalty_box_height / 2, field_dimensions))
    left_penalty_top_right = (penalty_box_width, invert_y(left_penalty_middle[1] - penalty_box_height / 2, field_dimensions))

    right_penalty_middle = (field_dimensions['x'], invert_y(field_dimensions['y'] / 2, field_dimensions))
    right_penalty_bottom_left = (field_dimensions['x'] - penalty_box_width, invert_y(right_penalty_middle[1] + penalty_box_height / 2, field_dimensions))
    right_penalty_bottom_right = (field_dimensions['x'], invert_y(right_penalty_middle[1] + penalty_box_height / 2, field_dimensions))
    right_penalty_top_left = (field_dimensions['x'] - penalty_box_width, invert_y(right_penalty_middle[1] - penalty_box_height / 2, field_dimensions))
    right_penalty_top_right = (field_dimensions['x'], invert_y(right_penalty_middle[1] - penalty_box_height / 2, field_dimensions))

    penalty_spot_left = (penalty_spot_distance_to_goal_line, invert_y(field_dimensions['y'] / 2, field_dimensions))
    penalty_spot_right = (field_dimensions['x'] - penalty_spot_distance_to_goal_line, invert_y(field_dimensions['y'] / 2, field_dimensions))

    print(
        {
            'pitch_corner': [top_left, top_right], # Model only trained to see top corners
            'penalty_area_corner': [left_penalty_top_right, left_penalty_bottom_right, right_penalty_top_left, right_penalty_bottom_left],
            'halfway_line_intersection': [top_middle],
            'center_circle_point': [middle],
            'penalty_spot': [penalty_spot_left, penalty_spot_right]
        },
    )

    # Lines
    l1 = (top_left, top_right)
    l2 = (top_right, bottom_right)
    l3 = (bottom_right, bottom_left)
    l4 = (bottom_left, top_left)
    l5 = (top_middle, bottom_middle)
    l6 = (left_penalty_top_left, left_penalty_top_right)
    l7 = (left_penalty_top_right, left_penalty_bottom_right)
    l8 = (left_penalty_bottom_right, left_penalty_bottom_left)
    l9 = (right_penalty_top_right, right_penalty_top_left)
    l10 = (right_penalty_top_left, right_penalty_bottom_left)
    l11 = (right_penalty_bottom_left, right_penalty_bottom_right)

    field_model = {
        "lines": [l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11],
        "circle": {"center": middle, "radius": circle_radius}
    }

    field_image = np.zeros((image_height, image_width), dtype=np.uint8)  # note: (H, W) shape

    def model2pix(pt, invert_y=False):
        x_m, y_m = pt
        x_p = x_m / field_dimensions['x'] * image_width
        # Invert Y axis: model (0,0) = bottom-left -> image (0,0) = top-left

        if (invert_y):
            y_p = (1.0 - y_m / field_dimensions['y']) * image_height
        else:
            y_p =  y_p = y_m / field_dimensions['y'] * image_height

        return int(round(x_p)), int(round(y_p))

    # Draw field lines
    for (p1, p2) in field_model['lines']:
        cv2.line(field_image, model2pix(p1), model2pix(p2), color=255, thickness=line_thickness)

    # Draw center circle
    cx, cy = field_model['circle']['center']
    r = field_model['circle']['radius']
    center_px = model2pix((cx, cy))
    radius_px = int(round(r / field_dimensions['x'] * image_width))  # assume scaling with x
    cv2.circle(field_image, center_px, radius_px, color=255, thickness=line_thickness)

    # Draw penalty spots
    penalty_spot_px_left = model2pix(penalty_spot_left)
    penalty_spot_px_right = model2pix(penalty_spot_right)
    cv2.circle(field_image, penalty_spot_px_left, radius = 2, color=255, thickness=line_thickness)
    cv2.circle(field_image, penalty_spot_px_right, radius = 2, color=255, thickness=line_thickness)

    _, field_image = cv2.threshold(field_image, 127, 255, cv2.THRESH_BINARY)

    save_path = os.path.join(os.getcwd(), "field.jpg")
    cv2.imwrite(save_path, field_image)

    return field_image

def generate_color_field_image(field_dimensions, image_width, image_height, line_thickness=2, scale=1.0, save_path=None):
    fw = int(round(image_width  * scale))
    fh = int(round(image_height * scale))

    # Points
    bottom_left = (0.0, invert_y(0, field_dimensions))
    bottom_right = (field_dimensions['x'], invert_y(0, field_dimensions))
    top_left = (0.0, invert_y(field_dimensions['y'], field_dimensions))
    top_right = (field_dimensions['x'], invert_y(field_dimensions['y'], field_dimensions))

    top_middle = (field_dimensions['x'] / 2, invert_y(field_dimensions['y'], field_dimensions))
    bottom_middle = (field_dimensions['x'] / 2, invert_y(0, field_dimensions))
    middle = (field_dimensions['x'] / 2, invert_y(field_dimensions['y'] / 2, field_dimensions))
    circle_radius = 9.15

    penalty_box_height = (2 * 11) + (2 * 5.5) + 7.32
    penalty_box_width = 5.5 + 11
    penalty_spot_distance_to_goal_line = 11

    left_penalty_middle = (0, invert_y(field_dimensions['y'] / 2, field_dimensions))
    left_penalty_bottom_left = (0, invert_y(left_penalty_middle[1] + penalty_box_height / 2, field_dimensions))
    left_penalty_bottom_right = (penalty_box_width, invert_y(left_penalty_middle[1] + penalty_box_height / 2, field_dimensions))
    left_penalty_top_left = (0, invert_y(left_penalty_middle[1] - penalty_box_height / 2, field_dimensions))
    left_penalty_top_right = (penalty_box_width, invert_y(left_penalty_middle[1] - penalty_box_height / 2, field_dimensions))

    right_penalty_middle = (field_dimensions['x'], invert_y(field_dimensions['y'] / 2, field_dimensions))
    right_penalty_bottom_left = (field_dimensions['x'] - penalty_box_width, invert_y(right_penalty_middle[1] + penalty_box_height / 2, field_dimensions))
    right_penalty_bottom_right = (field_dimensions['x'], invert_y(right_penalty_middle[1] + penalty_box_height / 2, field_dimensions))
    right_penalty_top_left = (field_dimensions['x'] - penalty_box_width, invert_y(right_penalty_middle[1] - penalty_box_height / 2, field_dimensions))
    right_penalty_top_right = (field_dimensions['x'], invert_y(right_penalty_middle[1] - penalty_box_height / 2, field_dimensions))

    penalty_spot_left = (penalty_spot_distance_to_goal_line, invert_y(field_dimensions['y'] / 2, field_dimensions))
    penalty_spot_right = (field_dimensions['x'] - penalty_spot_distance_to_goal_line, invert_y(field_dimensions['y'] / 2, field_dimensions))

    # Lines
    l1 = (top_left, top_right)
    l2 = (top_right, bottom_right)
    l3 = (bottom_right, bottom_left)
    l4 = (bottom_left, top_left)
    l5 = (top_middle, bottom_middle)
    l6 = (left_penalty_top_left, left_penalty_top_right)
    l7 = (left_penalty_top_right, left_penalty_bottom_right)
    l8 = (left_penalty_bottom_right, left_penalty_bottom_left)
    l9 = (right_penalty_top_right, right_penalty_top_left)
    l10 = (right_penalty_top_left, right_penalty_bottom_left)
    l11 = (right_penalty_bottom_left, right_penalty_bottom_right)

    lines = [l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11]
    circle_r_px = int(round((circle_radius / field_dimensions['x']) * fw))

    def model2pix(pt, invert_y=False):
        x_m, y_m = pt
        x_p = x_m / field_dimensions['x'] * image_width
        # Invert Y axis: model (0,0) = bottom-left -> image (0,0) = top-left

        if (invert_y):
            y_p = (1.0 - y_m / field_dimensions['y']) * image_height
        else:
            y_p =  y_p = y_m / field_dimensions['y'] * image_height

        return int(round(x_p)), int(round(y_p))

    img = np.full((image_height, image_width, 3), (0,100,0), dtype=np.uint8)
    for p1, p2 in lines:
        cv2.line(img,
                 model2pix(p1),
                 model2pix(p2),
                 color=(255,255,255),
                 thickness=line_thickness)
    cv2.circle(img,
               model2pix(middle),
               circle_r_px,
               color=(255,255,255),
               thickness=line_thickness)
    for spot in (penalty_spot_left, penalty_spot_right):
        cv2.circle(img,
                   model2pix(spot),
                   radius=line_thickness,
                   color=(255,255,255),
                   thickness=-1)

    if save_path is None:
        save_path = os.path.join(os.getcwd(), 'field.jpg')
        # ensure extension
    if not save_path.lower().endswith('.jpg'):
        save_path += '.jpg'
    cv2.imwrite(save_path, img)

    return img, save_path

def get_field_markers(field_dimensions):
    # Points
    bottom_left = (0.0, invert_y(0, field_dimensions))
    bottom_right = (field_dimensions['x'], invert_y(0, field_dimensions))
    top_left = (0.0, invert_y(field_dimensions['y'], field_dimensions))
    top_right = (field_dimensions['x'], invert_y(field_dimensions['y'], field_dimensions))

    top_middle = (field_dimensions['x'] / 2, invert_y(field_dimensions['y'], field_dimensions))
    middle = (field_dimensions['x'] / 2, invert_y(field_dimensions['y'] / 2, field_dimensions))

    penalty_box_height = (2 * 11) + (2 * 5.5) + 7.32
    penalty_box_width = 5.5 + 11
    penalty_spot_distance_to_goal_line = 11
    arc_radius = 9.15

    left_penalty_middle = (0, invert_y(field_dimensions['y'] / 2, field_dimensions))
    left_penalty_bottom_left = (0, invert_y(left_penalty_middle[1] + penalty_box_height / 2, field_dimensions))
    left_penalty_bottom_right = (penalty_box_width, invert_y(left_penalty_middle[1] + penalty_box_height / 2, field_dimensions))
    left_penalty_top_left = (0, invert_y(left_penalty_middle[1] - penalty_box_height / 2, field_dimensions))
    left_penalty_top_right = (penalty_box_width, invert_y(left_penalty_middle[1] - penalty_box_height / 2, field_dimensions))

    right_penalty_middle = (field_dimensions['x'], invert_y(field_dimensions['y'] / 2, field_dimensions))
    right_penalty_bottom_left = (field_dimensions['x'] - penalty_box_width, invert_y(right_penalty_middle[1] + penalty_box_height / 2, field_dimensions))
    right_penalty_bottom_right = (field_dimensions['x'], invert_y(right_penalty_middle[1] + penalty_box_height / 2, field_dimensions))
    right_penalty_top_left = (field_dimensions['x'] - penalty_box_width, invert_y(right_penalty_middle[1] - penalty_box_height / 2, field_dimensions))
    right_penalty_top_right = (field_dimensions['x'], invert_y(right_penalty_middle[1] - penalty_box_height / 2, field_dimensions))

    penalty_spot_left = (penalty_spot_distance_to_goal_line, invert_y(field_dimensions['y'] / 2, field_dimensions))
    penalty_spot_right = (field_dimensions['x'] - penalty_spot_distance_to_goal_line, invert_y(field_dimensions['y'] / 2, field_dimensions))

    top_central_circle = (field_dimensions['x'] / 2, invert_y(field_dimensions['y'] / 2 + arc_radius, field_dimensions))
    bottom_central_circle = (field_dimensions['x'] / 2, invert_y(field_dimensions['y'] / 2 - arc_radius, field_dimensions))

    left_arc_intersects = circle_rectangle_intersections([left_penalty_bottom_left, left_penalty_bottom_right, left_penalty_top_right, left_penalty_top_left], penalty_spot_left, arc_radius)
    right_arc_intersects = circle_rectangle_intersections([right_penalty_bottom_left, right_penalty_bottom_right, right_penalty_top_right, right_penalty_top_left], penalty_spot_right, arc_radius)

    left_penalty_arc_top = tuple(round(coord, 2) for coord in min(left_arc_intersects, key=lambda p: p[1]))
    left_penalty_arc_bottom = tuple(round(coord, 2) for coord in max(left_arc_intersects, key=lambda p: p[1]))
    right_penalty_arc_top = tuple(round(coord, 2) for coord in min(right_arc_intersects, key=lambda p: p[1]))
    right_penalty_arc_bottom = tuple(round(coord, 2) for coord in max(right_arc_intersects, key=lambda p: p[1]))

    synthetic_central_circle_right = ((field_dimensions['x'] / 2) + arc_radius, invert_y(field_dimensions['y'] / 2, field_dimensions))

    angle_degrees = 135
    angle_radians = math.radians(angle_degrees)
    r = arc_radius  # already defined in your function
    dx = r * math.cos(angle_radians)
    dy = r * math.sin(angle_radians)
    x = field_dimensions['x'] / 2 + dx
    y = field_dimensions['y'] / 2 + dy
    synthetic_topleft = (x, invert_y(y, field_dimensions))

    return {
        'middle_central_circle': [middle],
        'top_central_circle': [top_central_circle],
        'bottom_central_circle': [bottom_central_circle],
        'pitch_top_middle': [top_middle],
        'pitch_top_left_corner': [top_left],
        'left_penalty_box_top_corner': [left_penalty_top_right],
        'left_penalty_box_bottom_corner': [left_penalty_bottom_right],
        'left_penalty_arc_top': [left_penalty_arc_top],
        'left_penalty_arc_bottom': [left_penalty_arc_bottom],
        'right_penalty_arc_top': [right_penalty_arc_top],
        'right_penalty_arc_bottom': [right_penalty_arc_bottom],
        'pitch_top_right_corner': [top_right],
        'right_penalty_box_top_corner': [right_penalty_top_left],
        'right_penalty_box_bottom_corner': [right_penalty_bottom_left],
        'synthetic_central_circle_right': [synthetic_central_circle_right],
        'synthetic_central_circle_topleft': [synthetic_topleft]
    }

if __name__ == "__main__":
    field_dimensions = {'x': 105.0, 'y': 68.0}
    # image_scale = 8
    # generate_field_image(
    #     field_dimensions,
    #     image_width=image_scale * int(field_dimensions['x']),
    #     image_height=image_scale * int(field_dimensions['y'])
    # )
    generate_color_field_image(field_dimensions, 105 * 5, 68 * 5, 3, 10.0)
    print(get_field_markers(field_dimensions))

