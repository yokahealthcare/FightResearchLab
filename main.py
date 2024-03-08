import math
import time
from collections import deque

import cv2
from ultralytics import YOLO

TRAJECTORY_POINT = 30


class YoloPersonDetector:
    def __init__(self, yolo_model):
        self.model = YOLO(yolo_model)

    def run(self, source):
        return self.model.track(source, classes=[0], stream=True, tracker="bytetrack.yaml", persist=True)


def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


if __name__ == '__main__':
    yolo = YoloPersonDetector("asset/yolo/yolov8x.pt")

    prev_ids = None
    dict_of_ids = {}
    dict_of_velocities = {}
    dict_of_average_velocities = {}
    dict_of_variances = {}
    for result in yolo.run("asset/video/FIGHT_195_230.mp4"):
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
            mid_x, mid_y = int(mid_x), int(mid_y)

            # Append to deque
            dict_of_ids[_id].append([mid_x, mid_y])

        # Loop the trajectory points
        for _id, trajectory_points in dict_of_ids.items():
            length_of_trajectory_points = len(trajectory_points)

            for idx in range(length_of_trajectory_points):
                # Plot point of trajectories
                x = trajectory_points[idx][0]
                y = trajectory_points[idx][1]
                cv2.circle(result_frame, (x, y), radius=1, color=(255, 0, 0), thickness=2)

                # Calculate distance
                # Skip the last element & if it just one element present
                # Because we are going to calculate distance between two points (for last index will cause error)
                if idx == (length_of_trajectory_points - 1):
                    continue

                # In here distance == velocity
                x_next = trajectory_points[idx + 1][0]
                y_next = trajectory_points[idx + 1][1]
                velocity = calculate_distance(x, y, x_next, y_next)

                # Append to deque
                dict_of_velocities[_id].append(velocity)

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

        for (_id, box) in zip(ids, boxes):
            if len(dict_of_variances[_id]) == 0:
                continue

            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            if dict_of_variances[_id][0] < 0.35:
                cv2.putText(result_frame, f"FIGHT", (x1, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

        prev_ids = ids
        end = time.perf_counter()

        # Plot
        cv2.imshow("webcam", result_frame)
        # Wait for a key event and get the ASCII code
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
