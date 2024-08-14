from imutils.video import VideoStream
import numpy as np
import cv2
import imutils
import time

# Initialize the video stream from the webcam
vs = VideoStream(src=0).start()
time.sleep(2.0)  # Allow the camera to warm up

# Initialize the scores
score_own = 0
score_opponent = 0

# Define Goal Areas using lines (X, Y)
goal_line_own = 50
goal_line_opponent = 590

goal_cooldown = 50  # frames to wait before allowing another goal
cooldown_counter = 0

while True:
    # Capture frame-by-frame
    frame = vs.read()

    # If we did not grab a frame, then we have reached the end of the stream
    if frame is None:
        break

    # Resize the frame, blur it, and convert it to the HSV color space
    frame = imutils.resize(frame, width=640)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Create a mask for the color red (assuming the ball is red)
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Perform a series of erosions and dilations to remove any small blobs left in the mask
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Find contours in the mask and initialize the current (x, y) center of the ball
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center = None

    # Only proceed if at least one contour was found
    if len(contours) > 0:
        # Find the largest contour in the mask, then use it to compute the minimum enclosing circle and centroid
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # Only proceed if the radius meets a minimum size
        if radius > 10:
            # Draw the circle and centroid on the frame
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

            # Check if the ball is in the own goal area (left side)
            if center[0] < goal_line_own:
                score_opponent += 1
                cv2.putText(frame, "GOAL! for Opponent", (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.waitKey(1000)  # Wait for a second

            # Check if the ball is in the opponent's goal area (right side)
            elif center[0] > goal_line_opponent:
                score_own += 1
                cv2.putText(frame, "GOAL! for Own Team", (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.waitKey(1000)  # Wait for a second

    # Draw goal lines
    cv2.line(frame, (goal_line_own, 0), (goal_line_own, 480), (255, 0, 0), 2)
    cv2.line(frame, (goal_line_opponent, 0), (goal_line_opponent, 480), (0, 255, 0), 2)

    # Update and show the score
    score_text = f"Own: {score_own} Opponent: {score_opponent}"
    cv2.putText(frame, score_text, (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the frame to our screen
    cv2.imshow('Frame', frame)

    # If the `q` key is pressed, break from the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam pointer
vs.stop()

# Close all windows
cv2.destroyAllWindows()
