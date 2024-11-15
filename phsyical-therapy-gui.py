import cv2
import time
import math as m
import mediapipe as mp

def findDistance(x1, y1, x2, y2):
    dist = m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist

def findAngle(x1, y1, x2, y2):
    theta = m.acos((y2 - y1) * (-y1) / (m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * y1))
    degree = int(180 / m.pi) * theta
    return degree

def sendWarning():
    # Implement your warning mechanism (e.g., sound alert or message display)
    print("Warning: Bad posture detected for too long!")

good_frames = 0
bad_frames = 0

font = cv2.FONT_HERSHEY_SIMPLEX

blue = (255, 127, 0)
red = (50, 50, 255)
green = (127, 255, 0)
dark_blue = (127, 20, 0)
light_green = (127, 233, 100)
yellow = (0, 255, 255)
pink = (255, 0, 255)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

if __name__ == "__main__":
    
    # Change the index to 1 for external camera, or try other indices if needed
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    video_output = cv2.VideoWriter('output.mp4', fourcc, fps, frame_size)

    while True:
        
        success, image = cap.read()
        if not success:
            print("Skipping empty frame.")
            continue  # Continue to the next frame if the current one is empty

        # Convert the image color space
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        keypoints = pose.process(image_rgb)

        # Check if pose landmarks are detected
        if not keypoints.pose_landmarks:
            cv2.putText(image, "No pose detected", (10, 30), font, 0.9, red, 2)
            cv2.imshow('MediaPipe Pose', image)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
            continue

        lm = keypoints.pose_landmarks
        lmPose = mp_pose.PoseLandmark

        h, w = image.shape[:2]
        
        # Calculate relevant keypoints
        try:
            l_knee_x = int(lm.landmark[lmPose.LEFT_KNEE].x * w)
            l_knee_y = int(lm.landmark[lmPose.LEFT_KNEE].y * h)
            r_knee_x = int(lm.landmark[lmPose.RIGHT_KNEE].x * w)
            r_knee_y = int(lm.landmark[lmPose.RIGHT_KNEE].y * h)
            l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
            l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)
            r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
            r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)
            l_ankle_x = int(lm.landmark[lmPose.LEFT_ANKLE].x * w)
            l_ankle_y = int(lm.landmark[lmPose.LEFT_ANKLE].y * h)
            r_ankle_x = int(lm.landmark[lmPose.RIGHT_ANKLE].x * w)
            r_ankle_y = int(lm.landmark[lmPose.RIGHT_ANKLE].y * h)

            # Calculate body alignment and leg mobility based on visible side
            if lm.landmark[lmPose.LEFT_SHOULDER].visibility > lm.landmark[lmPose.RIGHT_SHOULDER].visibility:
                # Calculate for left side if left shoulder is more visible
                body_alignment = findAngle(l_shldr_x, l_shldr_y, l_knee_x, l_knee_y)
                leg_mobility = findAngle(l_knee_x, l_knee_y, l_ankle_x, l_ankle_y)
                cv2.putText(image, "Using Left Side", (10, 60), font, 0.7, green, 2)

            else:
                # Calculate for right side if right shoulder is more visible
                body_alignment = findAngle(r_shldr_x, r_shldr_y, r_knee_x, r_knee_y)
                leg_mobility = findAngle(r_knee_x, r_knee_y, r_ankle_x, r_ankle_y)
                cv2.putText(image, "Using Right Side", (10, 60), font, 0.7, green, 2)

            # Display the angles
            cv2.putText(image, f"Body Alignment: {int(body_alignment)} degrees", (10, 60), font, 0.9, green, 2)
            cv2.putText(image, f"Leg Mobility: {int(leg_mobility)} degrees", (10, 90), font, 0.9, blue, 2)

            angle_text_string = f'Body alignment : {int(body_alignment)}  Leg mobility : {int(leg_mobility)}'


            if (body_alignment < 190 and body_alignment > 170) and leg_mobility < 40:
                cv2.putText(image, str(int(body_alignment)) + str(int(leg_mobility)) + ' Aligned', (w - 150, 30), font, 0.9, green, 2)
                bad_frames = 0
                good_frames += 1
                color = light_green
            else:
                cv2.putText(image, str(int(body_alignment)) + str(int(leg_mobility)) + ' Not Aligned', (w - 150, 30), font, 0.9, red, 2)
                good_frames = 0
                bad_frames += 1
                color = red

            cv2.putText(image, angle_text_string, (10, 30), font, 0.9, color, 2)

            good_time = (1 / fps) * good_frames
            bad_time = (1 / fps) * bad_frames

            if bad_time > 180:
                sendWarning()

            if good_time > 0:
                cv2.putText(image, f'Good Posture Time: {round(good_time, 1)}s', (10, h - 20), font, 0.9, green, 2)
            else:
                cv2.putText(image, f'Bad Posture Time: {round(bad_time, 1)}s', (10, h - 20), font, 0.9, red, 2)

            # Draw landmarks and lines on the image
            cv2.circle(image, (l_shldr_x, l_shldr_y), 7, yellow, -1)
            cv2.circle(image, (l_knee_x, l_knee_y), 7, green, -1)
            cv2.circle(image, (l_ankle_x, l_ankle_y), 7, pink, -1)

            cv2.line(image, (l_shldr_x, l_shldr_y), (l_knee_x, l_knee_y), color, 4)
            cv2.line(image, (l_knee_x, l_knee_y), (l_ankle_x, l_ankle_y), color, 4)

        except Exception as e:
            print(f"Error: {e}")

        video_output.write(image)

        cv2.imshow('MediaPipe Pose', image)
        
        # Exit condition: Press 'q' to exit
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    video_output.release()
    cv2.destroyAllWindows()