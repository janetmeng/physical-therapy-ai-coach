import cv2
import math as m
import mediapipe as mp
import streamlit as st

# Helper functions for posture detection
def findDistance(x1, y1, x2, y2):
    return m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def findAngle(x1, y1, x2, y2):
    theta = m.acos((y2 - y1) * (-y1) / (m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * y1))
    degree = int(180 / m.pi) * theta
    return degree

#help function for 3 var
def findAngle3(x1, y1, x2, y2, x3, y3):
    # Calculate vectors and angle between them
    a, b = (x1 - x2, y1 - y2), (x3 - x2, y3 - y2)
    theta = m.acos((a[0] * b[0] + a[1] * b[1]) / (m.sqrt(a[0]**2 + a[1]**2) * m.sqrt(b[0]**2 + b[1]**2)))
    return theta * (180 / m.pi)


# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Streamlit UI
st.title("Ready-Set-Play")
camera_source = st.sidebar.selectbox("Select Camera Source", ["Default Camera", "External Camera 1"])
camera_index = 0 if camera_source == "Default Camera" else 1

#streamlit UI checkbox test:
# Title of the app
st.title("Which injury area would you like to assess?")

selected_option = st.selectbox(
    "Choose a feature",
    ["Squat", "Hamstring", "Arm T", "Arm I", "Arm Y"]
)

# Display the selected option
st.write(f"You selected: {selected_option}")

# Run Detection
run_detection = st.checkbox("Run Posture Detection")

if run_detection:
    good_frames = 0
    bad_frames = 0

    # Open video capture with selected camera index
    cap = cv2.VideoCapture(camera_index)

    # Set camera resolution to a more vertical (narrower) aspect ratio
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)  # Width smaller (narrower)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)  # Height larger (vertical)

    frame_window = st.image([])
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            st.warning("Unable to access camera.")
            break

        # Convert image for Mediapipe processing
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        keypoints = pose.process(image_rgb)

        # Check if pose landmarks are detected
        if keypoints.pose_landmarks:
            lm = keypoints.pose_landmarks
            lmPose = mp_pose.PoseLandmark

            h, w = image.shape[:2]
            
            if selected_option == "Hamstring":
            
                # Extract relevant keypoints
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

                    #new:
                    if lm.landmark[lmPose.LEFT_SHOULDER].visibility > lm.landmark[lmPose.RIGHT_SHOULDER].visibility:
                        # Calculate for left side if left shoulder is more visible
                        body_alignment = findAngle(l_shldr_x, l_shldr_y, l_knee_x, l_knee_y)
                        leg_mobility = findAngle(l_knee_x, l_knee_y, l_ankle_x, l_ankle_y)
                        cv2.putText(image, "Using Left Side", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

                    else:
                        # Calculate for right side if right shoulder is more visible
                        body_alignment = findAngle(r_shldr_x, r_shldr_y, r_knee_x, r_knee_y)
                        leg_mobility = findAngle(r_knee_x, r_knee_y, r_ankle_x, r_ankle_y)
                        cv2.putText(image, "Using Right Side", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)


                    if (body_alignment < 190 and body_alignment > 170) and leg_mobility < 50:
                        posture_text = f"Aligned: {int(body_alignment)}" 
                        color = (127, 255, 0)  # green
                    else:
                        posture_text = f"Not Aligned: {int(body_alignment)}" 
                        color = (50, 50, 255)  # red

                
                    if (body_alignment < 190 and body_alignment > 170) and leg_mobility < 50:
                        good_frames += 1
                        bad_frames = 0
                        color = (127, 233, 100)  # light green
                    else:
                        bad_frames += 1
                        good_frames = 0
                        color = (50, 50, 255)  # red

                    # Warning message if bad posture time exceeds 3 minutes (180 seconds)
                    if (1 / cap.get(cv2.CAP_PROP_FPS)) * bad_frames > 180:
                        st.warning("Warning: Poor posture detected for over 3 minutes!")

                    # Display feedback on the frame
                    cv2.putText(image, posture_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    cv2.putText(image, f"Body Alignment: {int(body_alignment)} | Leg Mobiliity: {int(leg_mobility)}", 
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    # Mark key points on the frame
                    cv2.circle(image, (l_shldr_x, l_shldr_y), 7, (0, 255, 255), -1)
                    cv2.circle(image, (l_knee_x, l_knee_y), 7, (0, 255, 0), -1)
                    cv2.circle(image, (l_ankle_x, l_ankle_y), 7, (203, 192, 255), -1)

                    cv2.line(image, (l_shldr_x, l_shldr_y), (l_knee_x, l_knee_y), color, 4)
                    cv2.line(image, (l_knee_x, l_knee_y), (l_ankle_x, l_ankle_y), color, 4)

                except Exception as e:
                    st.error(f"Error processing keypoints: {e}")

            if selected_option == "Squat":
                # Extract relevant keypoints
                try:
                    # Extract key points for left side (mirrored on right side if needed)
                    l_hip_x, l_hip_y = int(lm.landmark[lmPose.LEFT_HIP].x * w), int(lm.landmark[lmPose.LEFT_HIP].y * h)
                    r_hip_x, r_hip_y = int(lm.landmark[lmPose.RIGHT_HIP].x * w), int(lm.landmark[lmPose.RIGHT_HIP].y * h)

                    l_knee_x, l_knee_y = int(lm.landmark[lmPose.LEFT_KNEE].x * w), int(lm.landmark[lmPose.LEFT_KNEE].y * h)
                    r_knee_x, r_knee_y = int(lm.landmark[lmPose.RIGHT_KNEE].x * w), int(lm.landmark[lmPose.RIGHT_KNEE].y * h)

                    l_ankle_x, l_ankle_y = int(lm.landmark[lmPose.LEFT_ANKLE].x * w), int(lm.landmark[lmPose.LEFT_ANKLE].y * h)
                    r_ankle_x, r_ankle_y = int(lm.landmark[lmPose.RIGHT_ANKLE].x * w), int(lm.landmark[lmPose.RIGHT_ANKLE].y * h)

                    l_shldr_x, l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w), int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)
                    r_shldr_x, r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w), int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)


                    # Calculate and display angles if keypoints are detected

                    if lm.landmark[lmPose.LEFT_SHOULDER].visibility > lm.landmark[lmPose.RIGHT_SHOULDER].visibility:
                        # Calculate for left side if left shoulder is more visible
                        torso_angle = findAngle3(l_shldr_x, l_shldr_y, l_hip_x, l_hip_y, l_knee_x, l_knee_y)
                        knee_angle = findAngle3(l_hip_x, l_hip_y, l_knee_x, l_knee_y, l_ankle_x, l_ankle_y)
                        cv2.putText(image, "Using Left Side", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

                    else:
                        # Calculate for right side if right shoulder is more visible
                        torso_angle = findAngle3(r_shldr_x, r_shldr_y, r_hip_x, r_hip_y, r_knee_x, r_knee_y)
                        knee_angle = findAngle3(r_hip_x, r_hip_y, r_knee_x, r_knee_y, r_ankle_x, r_ankle_y)
                        cv2.putText(image, "Using Right Side", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)


                    # Define criteria for squat
                    squat_depth = 80 <= knee_angle <= 110
                    torso_upright = torso_angle > 70

                    # Determine squat position feedback
                    if squat_depth and torso_upright:
                        posture_text = "Good Squat Position"
                        color = (0, 255, 0)
                    else:
                        posture_text = "Incorrect Squat Position"
                        color = (0, 0, 255)

                    # Display feedback on frame
                    cv2.putText(image, posture_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    cv2.putText(image, f"Knee Angle: {int(knee_angle)} | Torso Angle: {int(torso_angle)}",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    # Mark key points
                    cv2.circle(image, (l_hip_x, l_hip_y), 7, (0, 255, 255), -1)
                    cv2.circle(image, (l_knee_x, l_knee_y), 7, (0, 255, 0), -1)
                    cv2.circle(image, (l_ankle_x, l_ankle_y), 7, (255, 0, 0), -1)
                    cv2.circle(image, (l_shldr_x, l_shldr_y), 7, (255, 255, 0), -1)

                    # Draw lines for visualization
                    cv2.line(image, (l_shldr_x, l_shldr_y), (l_hip_x, l_hip_y), color, 2)
                    cv2.line(image, (l_hip_x, l_hip_y), (l_knee_x, l_knee_y), color, 2)
                    cv2.line(image, (l_knee_x, l_knee_y), (l_ankle_x, l_ankle_y), color, 2)

                except Exception as e:  
                    st.error(f"Error processing keypoints: {e}")

            if selected_option == "Arm T":
                # Extract relevant keypoints
                try:
                    # Extract key points for left side (mirrored on right side if needed)
                    l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
                    l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)
                    r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
                    r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)

                    l_wrist_x = int(lm.landmark[lmPose.LEFT_WRIST].x * w)
                    l_wrist_y = int(lm.landmark[lmPose.LEFT_WRIST].y * h)
                    r_wrist_x = int(lm.landmark[lmPose.RIGHT_WRIST].x * w)
                    r_wrist_y = int(lm.landmark[lmPose.RIGHT_WRIST].y * h)
                    
                    # Calculate and display angles if keypoints are detected

                    if lm.landmark[lmPose.LEFT_SHOULDER].visibility > lm.landmark[lmPose.RIGHT_SHOULDER].visibility:
                        # Calculate for left side if left shoulder is more visible
                        arm_mobility = findAngle(l_shldr_x, l_shldr_y, l_wrist_x, l_wrist_y)
                        cv2.putText(image, "Using Left Side", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

                    else:
                        # Calculate for right side if right shoulder is more visible
                        arm_mobility = findAngle(r_shldr_x, r_shldr_y, r_wrist_x, r_wrist_y)
                        cv2.putText(image, "Using Right Side", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)                

                    # Check if the arm mobility is within the range -90 to 90 degrees
                    if 85 <= arm_mobility <= 95:
                        posture_text = "good T position"
                        color = (0, 255, 0)
                    else:
                        posture_text = "bad T position"
                        color = (0, 0, 255)

                    # Display feedback on frame
                    cv2.putText(image, posture_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    cv2.putText(image, f"Arm Mobility: {int(arm_mobility)}",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    if lm.landmark[lmPose.LEFT_SHOULDER].visibility > lm.landmark[lmPose.RIGHT_SHOULDER].visibility:
                    # Mark key points
                        cv2.circle(image, (l_shldr_x, l_shldr_y), 7, (0, 255, 255), -1)
                        cv2.circle(image, (l_wrist_x, l_wrist_y), 7, (0, 255, 0), -1)

                        # Draw lines for visualization
                        cv2.line(image, (l_shldr_x, l_shldr_y), (l_wrist_x, l_wrist_y), color, 4)
                    else:
                        cv2.circle(image, (r_shldr_x, r_shldr_y), 7, (0, 255, 255), -1)
                        cv2.circle(image, (r_wrist_x, r_wrist_y), 7, (0, 255, 0), -1)

                        # Draw lines for visualization
                        cv2.line(image, (r_shldr_x, r_shldr_y), (r_wrist_x, r_wrist_y), color, 4)


                except Exception as e:  
                    st.error(f"Error processing keypoints: {e}")

            if selected_option == "Arm I":
                # Extract relevant keypoints
                try:
                    # Extract key points for left side (mirrored on right side if needed)
                    l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
                    l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)
                    r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
                    r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)

                    l_wrist_x = int(lm.landmark[lmPose.LEFT_WRIST].x * w)
                    l_wrist_y = int(lm.landmark[lmPose.LEFT_WRIST].y * h)
                    r_wrist_x = int(lm.landmark[lmPose.RIGHT_WRIST].x * w)
                    r_wrist_y = int(lm.landmark[lmPose.RIGHT_WRIST].y * h)
                    
                    # Calculate and display angles if keypoints are detected

                    if lm.landmark[lmPose.LEFT_SHOULDER].visibility > lm.landmark[lmPose.RIGHT_SHOULDER].visibility:
                        # Calculate for left side if left shoulder is more visible
                        arm_mobility = findAngle(l_shldr_x, l_shldr_y, l_wrist_x, l_wrist_y)
                        cv2.putText(image, "Using Left Side", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

                    else:
                        # Calculate for right side if right shoulder is more visible
                        arm_mobility = findAngle(r_shldr_x, r_shldr_y, r_wrist_x, r_wrist_y)
                        cv2.putText(image, "Using Right Side", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)                

                    # Check if the arm mobility is within the range -90 to 90 degrees
                    if -5 <= arm_mobility <= 5:
                        posture_text = "good I position"
                        color = (0, 255, 0)
                    else:
                        posture_text = "bad I position"
                        color = (0, 0, 255)

                    # Display feedback on frame
                    cv2.putText(image, posture_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    cv2.putText(image, f"Arm Mobility: {int(arm_mobility)}",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    if lm.landmark[lmPose.LEFT_SHOULDER].visibility > lm.landmark[lmPose.RIGHT_SHOULDER].visibility:
                    # Mark key points
                        cv2.circle(image, (l_shldr_x, l_shldr_y), 7, (0, 255, 255), -1)
                        cv2.circle(image, (l_wrist_x, l_wrist_y), 7, (0, 255, 0), -1)

                        # Draw lines for visualization
                        cv2.line(image, (l_shldr_x, l_shldr_y), (l_wrist_x, l_wrist_y), color, 4)
                    else:
                        cv2.circle(image, (r_shldr_x, r_shldr_y), 7, (0, 255, 255), -1)
                        cv2.circle(image, (r_wrist_x, r_wrist_y), 7, (0, 255, 0), -1)

                        # Draw lines for visualization
                        cv2.line(image, (r_shldr_x, r_shldr_y), (r_wrist_x, r_wrist_y), color, 4)

                except Exception as e:  
                    st.error(f"Error processing keypoints: {e}")

            if selected_option == "Arm Y":
                # Extract relevant keypoints
                try:
                    # Extract key points for left side (mirrored on right side if needed)
                    l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
                    l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)
                    r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
                    r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)

                    l_wrist_x = int(lm.landmark[lmPose.LEFT_WRIST].x * w)
                    l_wrist_y = int(lm.landmark[lmPose.LEFT_WRIST].y * h)
                    r_wrist_x = int(lm.landmark[lmPose.RIGHT_WRIST].x * w)
                    r_wrist_y = int(lm.landmark[lmPose.RIGHT_WRIST].y * h)
                    
                    # Calculate and display angles if keypoints are detected

                    if lm.landmark[lmPose.LEFT_SHOULDER].visibility > lm.landmark[lmPose.RIGHT_SHOULDER].visibility:
                        # Calculate for left side if left shoulder is more visible
                        arm_mobility = findAngle(l_shldr_x, l_shldr_y, l_wrist_x, l_wrist_y)
                        cv2.putText(image, "Using Left Side", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

                    else:
                        # Calculate for right side if right shoulder is more visible
                        arm_mobility = findAngle(r_shldr_x, r_shldr_y, r_wrist_x, r_wrist_y)
                        cv2.putText(image, "Using Right Side", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)                

                    # Check if the arm mobility is within the range -90 to 90 degrees
                    if 40 <= arm_mobility <= 50:
                        posture_text = "good Y position"
                        color = (0, 255, 0)
                    else:
                        posture_text = "bad Y position"
                        color = (0, 0, 255)

                    # Display feedback on frame
                    cv2.putText(image, posture_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    cv2.putText(image, f"Arm Mobility: {int(arm_mobility)}",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    if lm.landmark[lmPose.LEFT_SHOULDER].visibility > lm.landmark[lmPose.RIGHT_SHOULDER].visibility:
                    # Mark key points
                        cv2.circle(image, (l_shldr_x, l_shldr_y), 7, (0, 255, 255), -1)
                        cv2.circle(image, (l_wrist_x, l_wrist_y), 7, (0, 255, 0), -1)

                        # Draw lines for visualization
                        cv2.line(image, (l_shldr_x, l_shldr_y), (l_wrist_x, l_wrist_y), color, 4)
                    else:
                        cv2.circle(image, (r_shldr_x, r_shldr_y), 7, (0, 255, 255), -1)
                        cv2.circle(image, (r_wrist_x, r_wrist_y), 7, (0, 255, 0), -1)

                        # Draw lines for visualization
                        cv2.line(image, (r_shldr_x, r_shldr_y), (r_wrist_x, r_wrist_y), color, 4)

                except Exception as e:  
                    st.error(f"Error processing keypoints: {e}")

        # Display the frame in Streamlit
        frame_window.image(image, channels="BGR")

    cap.release()