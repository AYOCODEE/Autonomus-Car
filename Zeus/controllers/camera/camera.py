"""camera controller with odometry-based turning."""

from controller import Robot, Camera
import tensorflow as tf
import numpy as np
import cv2
import threading
import queue
import math
import time

# create the Robot instance.
robot = Robot()
timestep = int(robot.getBasicTimeStep())

camera = robot.getDevice('Camera')
camera.enable(timestep)
camera.recognitionEnable(timestep)

leftInfra = robot.getDevice('left')
leftInfra.enable(timestep)

sonar = robot.getDevice('sonar')
sonar.enable(timestep)

# Motor setup
motors = []
motor_names = ['motor_1', 'motor_2', 'motor_3', 'motor_4']
for name in motor_names:
    motor = robot.getDevice(name)
    motor.setPosition(float('inf'))
    motor.setVelocity(0.0)
    motors.append(motor)

# Position sensor setup
left_ps = robot.getDevice('position_1')
right_ps = robot.getDevice('position_2')
left_ps.enable(timestep)
right_ps.enable(timestep)

# Odometry constants
wheel_radius = 0.06
dist_between_wheels = 0.240318
wheel_circum = 2 * math.pi * wheel_radius
encoder = wheel_circum / 6.28  # per radian



# Odometry turn function
def turn_angle(robot, angle_deg, left_ps, right_ps, motors, wheel_radius, dist_between_wheels, encoder, turn_speed):
    angle_rad = math.radians(angle_deg)
    turn_distance = (dist_between_wheels / 2.0) * abs(angle_rad)
    wheel_rotations = turn_distance / wheel_radius  # radians

    ps_start = [left_ps.getValue(), right_ps.getValue()]
    
    while robot.step(timestep) != -1:
        ps_current = [left_ps.getValue(), right_ps.getValue()]
        delta_left = abs(ps_current[0] - ps_start[0])
        delta_right = abs(ps_current[1] - ps_start[1])
        
        if delta_left >= wheel_rotations and delta_right >= wheel_rotations:
            break
        
        if angle_deg > 0:  # Left turn
            motors[0].setVelocity(turn_speed)
            motors[2].setVelocity(turn_speed)
            motors[1].setVelocity(-turn_speed)
            motors[3].setVelocity(-turn_speed)
        else:  # Right turn
            motors[0].setVelocity(-turn_speed)
            motors[2].setVelocity(-turn_speed)
            motors[1].setVelocity(turn_speed)
            motors[3].setVelocity(turn_speed)

    # Stop all motors
    for m in motors:
        m.setVelocity(0)

# Model and labels
model = tf.keras.models.load_model('mobilenetv2_finetuned_model.h5')
labels = ["100", "110", "120", "15", "20", "30", "40", "5", "50", "60", "70", "80", "90", "crosswalk", "left", "right", "stop"]

image_queue = queue.Queue()
result_queue = queue.Queue()

# Makes the prediction run on another thread to reduce wait time for processing image
def prediction_worker():
    while True:
        img = image_queue.get()
        if img is None:
            break
        predictions = model.predict(img)
        result_queue.put(predictions)

thread = threading.Thread(target=prediction_worker)
thread.start()

# Main loop
left_speed = 8
right_speed = 8
turning_speed = 0
previous_speed = 8
detected = False
stopped = False
last_stopped = False
last_cross = False


while robot.step(timestep) != -1:
    distance = leftInfra.getValue()
    # Grabbing left side of camera image for object recognition processing
    image = camera.getImage()
    np_image = np.frombuffer(image, np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4))
    np_image = cv2.cvtColor(np_image, cv2.COLOR_BGRA2RGB)
    
    height, width, _ = np_image.shape
    left_side = np_image[:, :width // 2]
    new_size = left_side.shape[1]
    center = height // 2
    half_new_size = new_size // 2
    start_y = max(center - half_new_size, 0)
    end_y = min(start_y + new_size, height)
    square_crop = left_side[start_y:end_y, :]
    
    img_resized = cv2.resize(square_crop, (400, 400)) / 255.0
    img_resized = np.expand_dims(img_resized, axis=0)
    
    # running prediction within infared sensor ray below 5
    if distance < 5:
        if not detected:
           detected = True
           image_queue.put(img_resized) 
            
    if not result_queue.empty():
        predictions = result_queue.get()
        lab1 = np.argmax(predictions)
        label = labels[lab1]
        print(f"Detected Sign: {label}")
        
        # Designated behavious for road signs
        match label:
            case "5":
                left_speed = right_speed = 4
                last_stopped = False
                last_cross = False
            case "15":
                left_speed = right_speed = 4.5
                last_stopped = False
                last_cross = False
            case "20":
                left_speed = right_speed = 5
                last_stopped = False
                last_cross = False
            case "30":
                left_speed = right_speed = 5.5
                last_stopped = False
                last_cross = False
            case "40":
                left_speed = right_speed = 6
                last_stopped = False
                last_cross = False
            case "50":
                left_speed = right_speed = 6.5
                last_stopped = False
                last_cross = False
            case "60":
                left_speed = right_speed = 7
                last_stopped = False
                last_cross = False
            case "70":
                left_speed = right_speed = 7.5
                last_stopped = False
                last_cross = False
            case "80":
                left_speed = right_speed = 8
                last_stopped = False
                last_cross = False
            case "90":
                left_speed = right_speed = 8.5
                last_stopped = False
                last_cross = False
            case "100":
                left_speed = right_speed = 9
                last_stopped = False
                last_cross = False
            case "110":
                left_speed = right_speed = 9.5
                last_stopped = False
                last_cross = False
            case "120":
                left_speed = right_speed = 10
                last_stopped = False
                last_cross = False
            case "crosswalk":
                if not last_cross:
                    print("Detected: Crosswalk — stopping")
                    previous_speed = left_speed
                    last_cross = True
                    last_stopped = False
                    left_speed = right_speed = 0
                    time.sleep(3)
                    
            # Calls odometry function for the left and right turning
            case "left":
                print("Detected: Left turn")
                turning_speed = left_speed
                last_stopped = False
                last_cross = False
                turn_angle(robot, 162, left_ps, right_ps, motors, wheel_radius, dist_between_wheels, encoder,turning_speed)
                left_speed = right_speed = turning_speed
            case "right":
                print("Detected: Right turn")
                turning_speed = left_speed
                last_stopped = False
                last_cross = False
                turn_angle(robot, -162, left_ps, right_ps, motors, wheel_radius, dist_between_wheels, encoder,turning_speed)
                left_speed = right_speed = turning_speed
            case "stop":
                if not last_stopped:
                    print("Detected: STOP sign — stopping")
                    previous_speed = left_speed
                    last_stopped = True
                    last_cross = False
                    left_speed = right_speed = 0
                    time.sleep(3)
                    left_speed = right_speed = previous_speed
            
    # Allows for second object detection
    if distance >= 5:
        detected = False
    # Detects obstacles   
    sonar_distance = sonar.getValue()
    if sonar_distance >= 5:
        left_speed = right_speed = previous_speed
    else:
        left_speed = right_speed = 0
        stop = int(5 * 1000 / timestep)
        for i in range(stop):
            left_speed = right_speed = 0
        

    
                
    # Apply motor speeds
    motors[0].setVelocity(right_speed)
    motors[2].setVelocity(right_speed)
    motors[1].setVelocity(left_speed)
    motors[3].setVelocity(left_speed)

    pass
# Stop worker thread on exit
image_queue.put(None)
thread.join()