# coding: utf-8
"""
Cody Nichoson
ECE 332
Introduction to Computer Vision
8 December, 2021

This code uses a Microsoft Kinect V2 to track a magic wand with an infrared reflective material
at its tip and draw a trace as it moves around the frame.
"""
import numpy as np
import cv2
import sys
from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, Registration, Frame

##################### KINECT V2 SETUP #####################
try:
    from pylibfreenect2 import OpenGLPacketPipeline
    pipeline = OpenGLPacketPipeline()
except:
    try:
        from pylibfreenect2 import OpenCLPacketPipeline
        pipeline = OpenCLPacketPipeline()
    except:
        from pylibfreenect2 import CpuPacketPipeline
        pipeline = CpuPacketPipeline()
print("Packet pipeline:", type(pipeline).__name__)

fn = Freenect2()
num_devices = fn.enumerateDevices()
if num_devices == 0:
    print("No device connected!")
    sys.exit(1)

serial = fn.getDeviceSerialNumber(0)
device = fn.openDevice(serial, pipeline=pipeline)

listener = SyncMultiFrameListener(FrameType.Color | FrameType.Ir | FrameType.Depth)

# Register listeners
device.setColorFrameListener(listener)
device.setIrAndDepthFrameListener(listener)

device.start()

# NOTE: must be called after device.start()
registration = Registration(device.getIrCameraParams(), device.getColorCameraParams())

undistorted = Frame(512, 424, 4)
registered = Frame(512, 424, 4)

# Optimal parameters for registration
# set True if you need
need_bigdepth = False
need_color_depth_map = False

bigdepth = Frame(1920, 1082, 4) if need_bigdepth else None
color_depth_map = np.zeros((424, 512),  np.int32).ravel() if need_color_depth_map else None

##################### INITIALIZE VARIBALES #####################
trace = [] # Initialize empty trace
cursor_color = (255, 255, 255, 1) # Initial trace color (white)
video_frames = []

# Experimenting with background subtractors
# fgbg1 = cv2.bgsegm.createBackgroundSubtractorMOG();  
# fgbg2 = cv2.createBackgroundSubtractorMOG2()
# fgbg3 = cv2.bgsegm.createBackgroundSubtractorGMG();

##################### MAIN VIDEO LOOP #####################
while True:
    # Get new frames from Kinect
    frames = listener.waitForNewFrame()

    # Separate various frames from Kinect inputs
    color = frames["color"]
    ir = frames["ir"]
    depth = frames["depth"]

    registration.apply(color, depth, undistorted, registered, bigdepth=bigdepth, color_depth_map=color_depth_map)

    ir_array = ir.asarray() / 65535.
    ir_blank = ir.asarray() / 65535.
    
    # Experimenting with background subtractors
    # fgmask1 = fgbg1.apply(ir_array)
    # fgmask2 = fgbg2.apply(ir_array)
    # fgmask3 = fgbg3.apply(ir_array)

    # Initilize empty mask array
    mask = np.zeros(ir_array.shape)

    # Get IR frame characteristics
    width = int(ir_array.shape[1])
    height = int(ir_array.shape[0])
    dim = (width, height)

    # Resize color array to match dimension of ir_array
    color_array = cv2.resize(color.asarray(), (int(1920 / 3), int(1080 / 3)))
    color_crop = color_array[:, 64:576]
    blank_edge = np.zeros((32, 512, 4), np.uint8)
    color_final = np.concatenate((blank_edge, color_crop, blank_edge), axis=0)

    # Values useful for rectangles
    center = 180
    height = 200
    gap = 8
    boxsize = 50
    fudge = 12

    # Define rectangle upper left corners (start points)
    # red = [center - int(gap/2) - boxsize - gap - boxsize, height]
    # blue = [center - int(gap/2) - boxsize, height+50]
    # yellow = [center + int(gap/2), height]
    # purple = [center + int(gap/2) + boxsize + gap, height]

    red = [512 - gap - boxsize, center - int(gap/2) - boxsize - gap - boxsize]
    blue = [512 - gap - boxsize, center - int(gap/2) - boxsize]
    yellow = [512 - gap - boxsize, center + int(gap/2)]
    purple = [512 - gap - boxsize, center + int(gap/2) + boxsize + gap]

    # Draw rectangles on color image
    cv2.rectangle(color_final, (red[0], red[1]), (red[0]+boxsize, red[1]+boxsize), (0, 0, 255), -1)
    cv2.rectangle(color_final, (yellow[0], yellow[1]), (yellow[0]+boxsize, yellow[1]+boxsize), (0, 255, 255), -1)
    cv2.rectangle(color_final, (blue[0], blue[1]), (blue[0]+boxsize, blue[1]+boxsize), (255, 0, 0), -1)
    cv2.rectangle(color_final, (purple[0], purple[1]), (purple[0]+boxsize, purple[1]+boxsize), (255, 20, 120), -1)
    
    # Find brightest pixel in IR image
    max = np.max(ir_array)

    if max > 0.99: # If the wand tip is in frame
        max_indices = np.where(ir_array == max)
        max_index = [max_indices[0][1], max_indices[1][1]]
        if max_index not in trace:
            trace.append(max_index)
        else:
            pass

        if len(trace) == 100: # or new_length == old_length:
            trace.pop(0)
    elif max < 0.99 and len(trace) > 0: # If wand tip not in frame, but trace exists
        trace.pop(0)
    elif max < 0.99: # If wand tip not in frame and no trace exists
        pass

    # Color changing - if wand tip within bounding box of color square, change cursor_color
    if max_index[0] > red[1]-gap and max_index[0] < red[1]+boxsize-gap and max_index[1] > red[0]-boxsize+gap and max_index[1] < red[0]:
        cursor_color = (0, 0, 255, 1)
    elif max_index[0] > blue[1] - gap and max_index[0] < blue[1] + boxsize - gap and max_index[1] > blue[0]-boxsize and max_index[1] < blue[0]:
        cursor_color = (255, 0, 0, 1)
    elif max_index[0] > yellow[1] - gap and max_index[0] < yellow[1] + boxsize - gap and max_index[1] > yellow[0]-boxsize and max_index[1] < yellow[0]:
        cursor_color = (0, 255, 255, 1)
    elif max_index[0] > purple[1] - gap and max_index[0] < purple[1] + boxsize - gap and max_index[1] > purple[0]-boxsize and max_index[1] < purple[0]:
        cursor_color = (255, 20, 120, 1)

    # Draw points in mask frame
    for point in trace:
        size = 3
        for i in range(point[0]-size, point[0]+size):
            for j in range(point[1]-size, point[1]+size):
                ir_array[i-1][j-1] = 255
                mask[i-1][j-1] = 255
                if i > 413:
                    i = 413
                if j > 471:
                    j = 471
                color_final[i+5][j+20] = cursor_color # Adding some offets to help with error between wand tip and trace

    # Display frames of interest
    cv2.imshow("Color", color_final)
    cv2.imshow("IR", ir_array)
    cv2.imshow("Mask", mask)

    listener.release(frames)

    key = cv2.waitKey(delay=1)
    if key == ord('q'):
        break

device.stop()
device.close()

sys.exit(0)