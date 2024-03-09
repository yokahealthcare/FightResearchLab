import math
import time
from collections import deque

import cv2
from ultralytics import YOLO

TRAJECTORY_POINT = 20


class YoloPersonDetector:
    def __init__(self, yolo_model):
        self.model = YOLO(yolo_model)

    def run(self, source):
        return self.model.track(source, classes=[0], stream=True, tracker="bytetrack.yaml", persist=True)


def calculate_distance(_x1, _y1, _x2, _y2):
    return math.sqrt(((_x2 - _x1) ** 2) + ((_y2 - _y1) ** 2))


def calculate_direction_of_motion(_x1, _y1, _x2, _y2):
    res = math.atan((_y2 - _y1) / (_x2 - _x1))
    res = math.degrees(res)

    """
        If the x-axis points to the right and the y-axis points down,
        this is a common convention in some fields such as computer graphics.
        However, it changes the way angles are measured.
        
        But the corrections would be different:
        If x2 is less than x1, you subtract 180° from the result.
        If the result is negative, you add 360° to it.
        
        These corrections ensure that the azimuth angle is within the range of 0 to 360°,
        with 0° pointing right, 90° pointing down, 180° pointing left, and 270° pointing up.
    """

    # if _x2 < _x1:
    #     res -= 180

    if res < 0:
        res += 360

    return res


if __name__ == '__main__':
    yolo = YoloPersonDetector("asset/yolo/yolov8x.pt")

    prev_ids = None
    dict_of_ids = {}
    dict_of_velocities = {}
    dict_of_direction_of_motion = {}
    dict_of_distribution_of_motion = {}
    dict_of_entropy_values = {}
    dict_of_history_entropy_values = {}
    dict_of_average_velocities = {}
    dict_of_variances = {}
    for result in yolo.run("asset/video/TEST_FIGHT_GLADAKAN_CUTTED.mp4"):
        start = time.perf_counter()

        original_frame = result.orig_img
        result_frame = result.plot()

        # Get ids
        ids = result.boxes.id
        if ids is None:  # Prevent error if nothing's detected
            continue

        # Motion Trajectory
        ids = [int(i) for i in ids.clone().tolist()]
        for _id in ids:
            if _id not in dict_of_ids:
                dict_of_ids[_id] = deque([], maxlen=TRAJECTORY_POINT)
                dict_of_velocities[_id] = deque([], maxlen=TRAJECTORY_POINT - 1)
                dict_of_direction_of_motion[_id] = deque([], maxlen=TRAJECTORY_POINT - 1)
                dict_of_distribution_of_motion[_id] = {
                    30: 0, 60: 0, 90: 0,
                    120: 0, 150: 0, 180: 0,
                    210: 0, 240: 0, 270: 0,
                    300: 0, 330: 0, 360: 0
                }
                dict_of_entropy_values[_id] = deque([], maxlen=1)
                dict_of_average_velocities[_id] = deque([], maxlen=1)
                dict_of_variances[_id] = deque([], maxlen=1)

        # Drop key if not present in scene
        if prev_ids is not None and ids != prev_ids:
            ids = set(ids)
            prev_ids = set(prev_ids)
            key_to_dump = prev_ids - ids
            for key in key_to_dump:
                dict_of_ids.pop(key)
                dict_of_velocities.pop(key)
                dict_of_direction_of_motion.pop(key)
                dict_of_distribution_of_motion.pop(key)
                dict_of_entropy_values.pop(key)
                dict_of_average_velocities.pop(key)
                dict_of_variances.pop(key)

        boxes = result.boxes.xyxy.tolist()
        for (_id, box) in zip(ids, boxes):
            x1, y1, x2, y2 = box
            # Find middle point in bounding box
            w = (x2 - x1) / 2
            h = (y2 - y1) / 2
            mid_x = x1 + w
            mid_y = y1 + h
            # mid_x, mid_y = int(mid_x), int(mid_y)

            # Append to deque
            dict_of_ids[_id].append([mid_x, mid_y])

        # Loop the trajectory points
        MIN_E = math.inf
        MAX_E = -math.inf
        for _id, trajectory_points in dict_of_ids.items():
            length_of_trajectory_points = len(trajectory_points)

            for idx in range(length_of_trajectory_points):
                # Plot point of trajectories
                x = trajectory_points[idx][0]
                y = trajectory_points[idx][1]
                cv2.circle(result_frame, (int(x), int(y)), radius=1, color=(255, 0, 0), thickness=2)

                # Calculate distance & direction of motion
                # Skip the last element & if it just one element present
                # Because we are going to calculate distance between two points (for last index will cause error)
                if idx == (length_of_trajectory_points - 1):
                    continue
                x_next = trajectory_points[idx + 1][0]
                y_next = trajectory_points[idx + 1][1]

                # In here distance == velocity
                velocity = calculate_distance(x, y, x_next, y_next)
                # Append to deque
                dict_of_velocities[_id].append(velocity)

                # Calculate direction of motion
                direction_of_motion = calculate_direction_of_motion(x, y, x_next, y_next)
                dict_of_direction_of_motion[_id].append(direction_of_motion)

            # Check if dict_of_velocities has been occupied or not
            length_of_velocities = len(dict_of_velocities[_id])
            if length_of_velocities < 2:
                continue

            # Calculate average velocity
            average_velocity = sum(dict_of_velocities[_id]) / len(dict_of_velocities[_id])
            # Append to deque
            dict_of_average_velocities[_id].append(average_velocity)

            # Calculate variance
            variance_square = 0
            for velocity in dict_of_velocities[_id]:
                variance_square += (velocity - average_velocity) ** 2

            variance = variance_square / (length_of_velocities - 1)
            dict_of_variances[_id].append(variance)

            # Calculate distribution of trajectory point of direction motions
            for direction_of_motion in dict_of_direction_of_motion[_id]:
                for degree, count in dict_of_distribution_of_motion[_id].items():
                    if direction_of_motion < degree:
                        dict_of_distribution_of_motion[_id][degree] += 1
                        break

            # Calculate Entropy Value
            entropy_value = 0
            for degree, count in dict_of_distribution_of_motion[_id].items():
                if count <= 0:
                    continue
                entropy_value = entropy_value + (count * math.log2(count))
            else:
                entropy_value = -entropy_value
            dict_of_entropy_values[_id].append(entropy_value)

            # Update the MAX_E (if true)
            if MAX_E < dict_of_entropy_values[_id][0]:
                MAX_E = dict_of_entropy_values[_id][0]
            # Update the MIN_E (if true)
            if MIN_E > dict_of_entropy_values[_id][0]:
                MIN_E = dict_of_entropy_values[_id][0]

            # Reset variable
            dict_of_distribution_of_motion[_id] = {
                30: 0, 60: 0, 90: 0,
                120: 0, 150: 0, 180: 0,
                210: 0, 240: 0, 270: 0,
                300: 0, 330: 0, 360: 0
            }

        # Normalize entropy values withing range 0 to 1
        for _id, trajectory_points in dict_of_ids.items():
            if len(dict_of_entropy_values[_id]) == 0:
                continue

            try:
                normalize_entropy_value = ((dict_of_entropy_values[_id][0]) - MIN_E) / (MAX_E - MIN_E)
                dict_of_entropy_values[_id].append(normalize_entropy_value)
            except ZeroDivisionError:
                pass

        for (_id, box) in zip(ids, boxes):
            if len(dict_of_variances[_id]) == 0:
                continue

            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.putText(result_frame, f"{dict_of_variances[_id][0]:.2f}", (x1, y1 + 60), cv2.FONT_HERSHEY_SIMPLEX,0.6, (255, 255, 255))
            if dict_of_variances[_id][0] > 10000:
                cv2.putText(result_frame, f"FIGHT", (x1, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

        prev_ids = ids
        end = time.perf_counter()

        # Plot
        cv2.imshow("webcam", result_frame)
        # Wait for a key event and get the ASCII code
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
