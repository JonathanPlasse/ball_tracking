from dataclasses import dataclass
from enum import Enum
from typing import Union

import cv2
import imutils
from imutils.video import VideoStream

HEIGHT = 750
WIDTH = 1000
FRAME_REBOUND = 5


class Vertical(Enum):
    TOP = "Top"
    DOWN = "Down"


class Horizontal(Enum):
    LEFT = "Left"
    RIGHT = "Right"


@dataclass
class Position:
    vertical: Vertical
    horizontal: Horizontal


def get_frame():
    return imutils.resize(vs.read(), width=1000)


def calculate_hsv(frame):
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    return cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)


def calculate_mask(hsv, lower_bound, upper_bound):
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    return mask


def find_center(mask, color):
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None
    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        if radius > 10:
            cv2.circle(frame, (int(x), int(y)), int(radius), color, 2)
    return center


def get_current_position(center) -> Position:
    vertical = Vertical.TOP if center[1] < HEIGHT // 2 else Vertical.DOWN
    horizontal = Horizontal.RIGHT if center[0] < WIDTH // 2 else Horizontal.LEFT
    return Position(vertical, horizontal)


def debound_position(position: Position) -> Position:
    nb_vertical_frame[position.vertical] += 1
    nb_horizontal_frame[position.horizontal] += 1

    for nb_frame in (nb_vertical_frame, nb_horizontal_frame):
        if min(nb_frame.items(), key=lambda d: d[1])[1] >= FRAME_REBOUND:
            nb_frame[max(nb_frame.items(), key=lambda d: d[1])[0]] = 0

    return Position(
        max(nb_vertical_frame, key=lambda v: nb_vertical_frame[v]),
        max(nb_horizontal_frame, key=lambda v: nb_horizontal_frame[v]),
    )


# [Yellow, Blue, Red, Green]
bounds = (0, 221, 80), (5, 255, 170)
color = (10, 40, 160)

vs = VideoStream(src=0).start()

nb_vertical_frame = {vertical: 0 for vertical in Vertical}
nb_horizontal_frame = {horizontal: 0 for horizontal in Horizontal}

while True:
    frame = get_frame()
    hsv = calculate_hsv(frame)

    mask = calculate_mask(hsv, *bounds)
    center = find_center(mask, color)

    if center is not None:
        position = get_current_position(center)
        print(position)
        position = debound_position(position)
        print(position)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

vs.stop()
cv2.destroyAllWindows()
