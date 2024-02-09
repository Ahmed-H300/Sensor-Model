import cv2
from typing import Tuple, List
from math import sin, cos, pi
from skimage.draw import line as get_points
import numpy as np
from numpy.linalg import norm
from enum import Enum
from joblib import Parallel, delayed

# define Colors


class Colors(Enum):
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)

# define class Robot


class Robot:
    # To define Constants
    def __init__(self,
                 sensor_range,
                 max_range_radius,
                 robot_radius,
                 robot_border_thickness,
                 robot_color,
                 sensor_resolution,
                 ray_color
                 ):
        # define the range of the sensor
        self.SENSOE_RANGE = sensor_range
        # define the resolution
        self.SENSOR_RESOLUTION = sensor_resolution
        # define the max range radius
        self.MAX_RANGE_RADIUS = max_range_radius
        # define the robot radius
        self.ROBOT_RADIUS = robot_radius
        # define the length of the direction line
        self.DIRECTION_LINE_LENGTH = 2 * robot_radius
        # define the robot border thickness
        self.ROBOT_BORDER_THICKNESS = robot_border_thickness
        # define the robot color
        self.ROBOT_COLOR = robot_color
        # define the color of the ray
        self.RAY_COLOR = ray_color
        # define the fill
        self.FILL = -1
        # define the colors
        self.BORDER_COLOR = (10, 10, 10)

    # convert degree to radian

    def rad(self, theta: float) -> float:
        # return theta in rad
        return theta * (pi / 180)

    # get the new location form x and y moving r with theta angle
    def get_location_form_xy_r_theta(self, x: int, y: int, r: float, theta: float) -> Tuple[int, int]:
        # get movement in x
        x2 = int(x + r * cos(self.rad(theta)))
        # get movement in y
        y2 = int(y + r * sin(self.rad(theta)))
        # return x2, y2
        return (x2, y2)

    # draw line on an image form start pixel to end pixel
    def draw_line(self, image: cv2.Mat, start_pixel: Tuple[int, int], length: float, angle: float, color: Tuple[float, float, float], thickness: int) -> None:
        # get location of end point
        end_point = self.get_location_form_xy_r_theta(
            start_pixel[0], start_pixel[1], length, angle)
        # draw the line
        cv2.line(image, start_pixel, end_point, color, thickness)

    # draw the laser ray and detect if hit the wall
    def draw_laser_line(self, image: cv2.Mat, center_pixel: Tuple[int, int], theta: float) -> None:
        # get end point of the ray
        end_point = self.get_location_form_xy_r_theta(
            center_pixel[0], center_pixel[1], self.MAX_RANGE_RADIUS, theta)
        # get all pixels between the max point and the center
        rr, cc = get_points(
            center_pixel[0], center_pixel[1], end_point[0], end_point[1])
        # list of points
        points = list(zip(rr, cc))
        # define final point
        final_point = end_point
        for x, y in points:
            # check if the point is black
            if norm(image[y, x] - np.array(Colors.BLACK.value)) == 0:
                # set final point
                final_point = (x, y)
                # break the loop
                break

        # draw the line
        cv2.line(image, center_pixel, final_point, self.RAY_COLOR, 1)

    # draw the robot circle
    def draw_robot(self, image: cv2.Mat, center_pixel: Tuple[int, int], start_theta: float) -> None:
        # draw circle on image
        image = cv2.circle(image, center_pixel, self.ROBOT_RADIUS,
                           self.ROBOT_COLOR, thickness=self.FILL)
        # draw border
        image = cv2.circle(image, center_pixel, self.ROBOT_RADIUS,
                           self.BORDER_COLOR, self.ROBOT_BORDER_THICKNESS)
        # draw line at the angle of start_theta
        self.draw_line(image, center_pixel, self.DIRECTION_LINE_LENGTH, angle=start_theta,
                       color=self.BORDER_COLOR, thickness=self.ROBOT_BORDER_THICKNESS)

    # draw laser range
    def draw_laser_range(self, image: cv2.Mat, center_pixel: Tuple[int, int], start_theta: float) -> None:
        # range for start_theta - (RANGE/2) to start_theta + (RANGE/2)
        for theta in range(int(start_theta - (self.SENSOE_RANGE/2)), int(start_theta + (self.SENSOE_RANGE/2)), 2):
            # draw the laser line
            self.draw_laser_line(image, center_pixel, theta)

# Requirement 1


def Req1():
    # Image path
    image_path = "Map.jpg"
    # Reading an image in color mode
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    # Start Position and angle
    start_position = (350, 185)
    start_angle = 200
    # define Robot specifications
    sensor_range = 250
    # max range in meters
    max_range_radius = 12
    # pixel size in cm
    pixel_size = 4
    # max_range in pixels
    max_range_radius = int(max_range_radius * 100 / pixel_size)
    # robot radius
    robot_radius = 10
    # robot border thickness
    robot_border_thickness = 2
    # robot color
    robot_color = Colors.GREEN.value
    # sensor resolution
    sensor_resolution = 2
    # ray color
    ray_color = Colors.RED.value
    # define the robot
    robot = Robot(
        sensor_range=sensor_range,
        max_range_radius=max_range_radius,
        robot_radius=robot_radius,
        robot_border_thickness=robot_border_thickness,
        robot_color=robot_color,
        sensor_resolution=sensor_resolution,
        ray_color=ray_color
    )
    # draw the robot
    robot.draw_robot(image, start_position, start_angle)
    # Write the image with robot
    image_name = 'map_with_robot.jpg'
    cv2.imwrite(image_name, image)
    print(
        f'Image after robot drawing is written with name {image_name} is written successfully')
    # uncomment this to show the image after robot drawing
    # # show the map with robot
    # cv2.imshow("Map_with_robot", image)
    # # wait for any key
    # cv2.waitKey(0)
    # # destroy all windows
    # cv2.destroyAllWindows()
    # draw traced laser rays on the input map
    robot.draw_laser_range(
        image=image,
        center_pixel=start_position,
        start_theta=start_angle
    )
    # Write the image with rays
    image_name = 'map_with_rays.jpg'
    cv2.imwrite(image_name, image)
    print(
        f'Image after rays drawing is written with name {image_name} is written successfully')
    # uncomment this to show the image after rays drawing
    # # show the map with traced rays
    # cv2.imshow("Map_with_rays", image)
    # # wait for any key
    # cv2.waitKey(0)
    # # destroy all windows
    # cv2.destroyAllWindows()


class EndPoint:
    # init
    def __init__(self):
        # pixel size in cm
        self.pixel_size = 4
        self.m_cm = 100

    # convert degree to radian
    def rad(self, theta: float) -> float:
        # return theta in rad
        return theta * (pi / 180)

    # get the new location form x and y moving r with theta angle
    def get_location_form_xy_r_theta(self, x: int, y: int, r: float, theta: float) -> Tuple[int, int]:
        # get movement in x
        x2 = int(x + r * cos(self.rad(theta)))
        # get movement in y
        y2 = int(y + r * sin(self.rad(theta)))
        # return x2, y2
        return (x2, y2)

    # function take image to calculate the likelihood field
    def calculate_likelihood_field(self, distances_filed: np.ndarray, sigma: int) -> np.ndarray:
        # calculate the likelihood field
        likelihood_field = 1/(np.sqrt(2*pi)*sigma) * np.exp(-0.5*((distances_filed)/sigma)**2)

        # max likelihood
        likelihood_max_value = np.max(likelihood_field)

        # Normalize all values to be between 0 and 1
        likelihood_field = likelihood_field/likelihood_max_value

        # # uncomment this to show the  likelihood image
        # # show the map
        # cv2.imshow(f"likelihood_field_sigma_{sigma}", likelihood_field*255)
        # # wait for any key
        # cv2.waitKey(0)
        # # destroy all windows
        # cv2.destroyAllWindows()
        # return

        return likelihood_field

    # function that takes likelihood_field and distance to calculate the probability map
    def calculate_probability_map(self, likelihood_field: np.ndarray, distance: int) -> np.ndarray:
        # calculate distance in pixels
        distance_pixel = int(distance * self.m_cm / self.pixel_size)
        # create an array to save the probabilities
        propbability_map = np.zeros(
            shape=(len(likelihood_field), len(likelihood_field[0])))
        # add the random value with a uniform distribution with probability 0.1
        # 300 to make sure it is over than 1
        # get max value
        likelihood_max_value = np.max(likelihood_field)
        likelihood_field = 0.9 * likelihood_field + \
            0.1 / (300 * likelihood_max_value)
        # calculate the probability map
        angle_step = 1
        probability_out_map = 0.000001
        # loop over all points in the map
        for y in range(0, likelihood_field.shape[0], 1):
            for x in range(0, likelihood_field.shape[1], 1):
                for theta in range(0, 360, angle_step):
                    # calculate probability
                    probability = 1
                    # get end point of the ray
                    end_point = self.get_location_form_xy_r_theta(
                        x, y, distance_pixel, theta)
                    # Check if it is out of map
                    if end_point[0] < 0 or end_point[0] >= likelihood_field.shape[1] or end_point[1] < 0 or end_point[1] >= likelihood_field.shape[0]:
                        probability *= probability_out_map
                    # if it is max radius range
                    else:
                        probability *= likelihood_field[end_point[1]
                                                        ][end_point[0]]

                    # update the probability map if the probability is higher
                    propbability_map[y][x] = max(
                        propbability_map[y][x], probability)

        return propbability_map

    def calculate_probability_for_point(self, x, y, likelihood_field, distance_pixel, probability_out_map):
        probabilities = []
        angle_step = 1
        # Increase the step size to reduce the number of iterations
        for theta in range(0, 360, angle_step):
            end_point = self.get_location_form_xy_r_theta(
                x, y, distance_pixel, theta)
            if end_point[0] < 0 or end_point[0] >= likelihood_field.shape[1] or end_point[1] < 0 or end_point[1] >= likelihood_field.shape[0]:
                probabilities.append(probability_out_map)
            else:
                probabilities.append(
                    likelihood_field[end_point[1]][end_point[0]])
        return max(probabilities)

    def calculate_probability_map_parallel(self, likelihood_field: np.ndarray, distance: int) -> np.ndarray:
        # calculate distance in pixels
        distance_pixel = int(distance * self.m_cm / self.pixel_size)
        likelihood_max_value = np.max(likelihood_field)
        likelihood_field = 0.9 * likelihood_field + \
            0.1 / (300 * likelihood_max_value)
        probability_out_map = 0.000001

        probability_map = Parallel(n_jobs=-1)(delayed(self.calculate_probability_for_point)(x, y, likelihood_field, distance_pixel,
                                                                                            probability_out_map) for y in range(likelihood_field.shape[0]) for x in range(likelihood_field.shape[1]))
        return np.array(probability_map).reshape(likelihood_field.shape)


# Requirement 2
def Req2():
    # Image path
    image_path = "Map.jpg"
    # image in grey scale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # do binary thresholding
    binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)[1]
    # Apply the distance transform to obtain distance to the nearst obstacle
    distances_filed = cv2.distanceTransform(binary_image, cv2.DIST_L2, 0)
    # pass the likelihood field through the gaussian function 
    distances_filed = np.array(distances_filed)
    ###########
    distances = [0.1, 0.3, 0.5, 1, 3, 5, 10]
    sigmas = [1, 5, 10, 15, 20, 25]
    # loop over sigmas and distances
    for sigma in sigmas:
        # get likely hoot
        likelihood_field = EndPoint().calculate_likelihood_field(distances_filed, sigma)
        # output the likelihood field
        likelihood_image = likelihood_field * 255
        # write the image
        cv2.imwrite(
            f"likelihood_field_sigma_{sigma}.jpg", likelihood_image)
        print(f"likelihood_field_sigma_{sigma}.jpg is written successfully")
        for distance in distances:
            # get the probablility map
            probability_map = EndPoint().calculate_probability_map_parallel(
                likelihood_field, distance)
            # output the probability map
            probability_image = probability_map * 255
            # write the image
            cv2.imwrite(
                f"propbability_map_sigma_{sigma}_distance_{distance}.jpg", probability_image)
            print(
                f"propbability_map_sigma_{sigma}_distance_{distance}.jpg is written successfully")


if __name__ == "__main__":
    print('***********************')
    # Req1
    print('Req1 started')
    Req1()
    print('Req1 finished')
    # Req2
    print('Req2 started')
    Req2()
    print('Req2 finished')
    print('***********************')
