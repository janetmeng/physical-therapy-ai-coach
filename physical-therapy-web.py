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

done_with_exercise = False

# Function to extract key points from landmarks
def extract_keypoints(lm, lmPose, side, w, h):
    if side == "left":
        shoulder_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
        shoulder_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)
        wrist_x = int(lm.landmark[lmPose.LEFT_WRIST].x * w)
        wrist_y = int(lm.landmark[lmPose.LEFT_WRIST].y * h)
    elif side == "right":
        shoulder_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
        shoulder_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)
        wrist_x = int(lm.landmark[lmPose.RIGHT_WRIST].x * w)
        wrist_y = int(lm.landmark[lmPose.RIGHT_WRIST].y * h)
    return shoulder_x, shoulder_y, wrist_x, wrist_y

# Function to calculate arm mobility (angle)
def calculate_arm_mobility(shoulder_x, shoulder_y, wrist_x, wrist_y):
    return findAngle(shoulder_x, shoulder_y, wrist_x, wrist_y)

# Function to determine the posture based on arm mobility
def check_posture(arm_mobility, position_type):
    if position_type == "T":
        if 85 <= arm_mobility <= 95:
            done_with_exercise = True
            return "good T position", (0, 255, 0)

        else:
            done_with_exercise = False
            return "bad T position", (0, 0, 255)

    elif position_type == "I":
        if -5 <= arm_mobility <= 5:
            done_with_exercise = True
            return "good I position", (0, 255, 0)
        else:
            done_with_exercise = False
            return "bad I position", (0, 0, 255)
    elif position_type == "Y":
        if 40 <= arm_mobility <= 50:
            done_with_exercise = True
            return "good Y position", (0, 255, 0)
        else:
            done_with_exercise = False
            return "bad Y position", (0, 0, 255)

# Function to display feedback and draw markers
def display_feedback(image, shoulder_x, shoulder_y, wrist_x, wrist_y, arm_mobility, posture_text, color):
    # Display posture and mobility info
    cv2.putText(image, posture_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(image, f"Arm Mobility: {int(arm_mobility)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    # Mark key points
    cv2.circle(image, (shoulder_x, shoulder_y), 7, (0, 255, 255), -1)
    cv2.circle(image, (wrist_x, wrist_y), 7, (0, 255, 0), -1)
    # Draw lines for visualization
    cv2.line(image, (shoulder_x, shoulder_y), (wrist_x, wrist_y), color, 4)

# Main function to process different arm positions
def process_arm_position(selected_option, lm, lmPose, w, h, image):
    try:
        if "Left Arm" in selected_option:
            side = "left"
        elif "Right Arm" in selected_option:
            side = "right"
        
        position_type = selected_option.split()[-1]  # "T", "I", or "Y"
        
        # Extract keypoints
        shoulder_x, shoulder_y, wrist_x, wrist_y = extract_keypoints(lm, lmPose, side, w, h)
        
        # Calculate arm mobility
        arm_mobility = calculate_arm_mobility(shoulder_x, shoulder_y, wrist_x, wrist_y)
        
        # Get posture feedback
        posture_text, color = check_posture(arm_mobility, position_type)
        
        # Display feedback and draw markers
        display_feedback(image, shoulder_x, shoulder_y, wrist_x, wrist_y, arm_mobility, posture_text, color)
        
    except Exception as e:
        st.error(f"Error processing keypoints: {e}")

#variables
exercises = []
done_tracker = []

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Streamlit UI
st.title("RecoverFit Pro")
camera_source = st.sidebar.selectbox("Select Camera Source", ["Default Camera", "External Camera 1"])
camera_index = 0 if camera_source == "Default Camera" else 1

st.subheader("Which injury area would you like to assess?")

selected_option = st.selectbox(
    "Choose an exercise",
    ["Squat", "Left Quad", "Right Quad", "Left Arm T", "Right Arm T", "Left Arm I", "Right Arm I", "Left Arm Y", "Right Arm Y"]
)

# Display the selected option
st.write(f"You selected: {selected_option}")
    
# Run Detection, initally false
run_detection = st.checkbox("Run Exercise Detection. Uncheck when done.",value = False)

# Create a button that resets the checkbox to unchecked when clicked
# if st.button("Reset Checkbox"):
#     run_detection = False  # Uncheck the box manually when the button is clicked

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
        
            #QUADS     
            if selected_option == "Left Quad":
                # Extract relevant keypoints
                try:
                    l_knee_x = int(lm.landmark[lmPose.LEFT_KNEE].x * w)
                    l_knee_y = int(lm.landmark[lmPose.LEFT_KNEE].y * h)
                    l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
                    l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)
                    l_ankle_x = int(lm.landmark[lmPose.LEFT_ANKLE].x * w)
                    l_ankle_y = int(lm.landmark[lmPose.LEFT_ANKLE].y * h)
                    
                    body_alignment = findAngle(l_shldr_x, l_shldr_y, l_knee_x, l_knee_y)
                    leg_mobility = findAngle(l_knee_x, l_knee_y, l_ankle_x, l_ankle_y)
                    cv2.putText(image, "Using Left Side", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
                  
                    if (body_alignment < 190 and body_alignment > 170) and leg_mobility < 50:
                        posture_text = f"Aligned: {int(body_alignment)}" 
                        color = (127, 255, 0)  # green
                        done_with_exercise = True
                    else:
                        posture_text = f"Not Aligned: {int(body_alignment)}" 
                        color = (50, 50, 255)  # red
                        done_with_exercise = False

                    # Display feedback on the frame
                    cv2.putText(image, posture_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    cv2.putText(image, f"Body Alignment: {int(body_alignment)} | Leg Mobiliity: {int(leg_mobility)}", 
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    #Draw points
                    cv2.circle(image, (l_shldr_x, l_shldr_y), 7, (0, 255, 255), -1)
                    cv2.circle(image, (l_knee_x, l_knee_y), 7, (0, 255, 0), -1)
                    cv2.circle(image, (l_ankle_x, l_ankle_y), 7, (203, 192, 255), -1)

                    # Draw lines for visualization
                    cv2.line(image, (l_shldr_x, l_shldr_y), (l_knee_x, l_knee_y), color, 4)
                    cv2.line(image, (l_knee_x, l_knee_y), (l_ankle_x, l_ankle_y), color, 4)
                   
                except Exception as e:
                    st.error(f"Error processing keypoints: {e}")
            if selected_option == "Right Quad":
                try:
                    r_knee_x = int(lm.landmark[lmPose.RIGHT_KNEE].x * w)
                    r_knee_y = int(lm.landmark[lmPose.RIGHT_KNEE].y * h)
                    r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
                    r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)
                    r_ankle_x = int(lm.landmark[lmPose.RIGHT_ANKLE].x * w)
                    r_ankle_y = int(lm.landmark[lmPose.RIGHT_ANKLE].y * h)
                 
                    # Calculate for right side if right shoulder is more visible
                    body_alignment = findAngle(r_shldr_x, r_shldr_y, r_knee_x, r_knee_y)
                    leg_mobility = findAngle(r_knee_x, r_knee_y, r_ankle_x, r_ankle_y)
                    cv2.putText(image, "Using Right Side", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

                    if (body_alignment < 190 and body_alignment > 170) and leg_mobility < 50:
                        posture_text = f"Aligned: {int(body_alignment)}" 
                        color = (127, 255, 0)  # green
                        done_with_exercise = True
                    else:
                        posture_text = f"Not Aligned: {int(body_alignment)}" 
                        color = (50, 50, 255)  # red
                        done_with_exercise = False


                    # Display feedback on the frame
                    cv2.putText(image, posture_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    cv2.putText(image, f"Body Alignment: {int(body_alignment)} | Leg Mobiliity: {int(leg_mobility)}", 
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    #Draw points
                    cv2.circle(image, (r_shldr_x, r_shldr_y), 7, (0, 255, 255), -1)
                    cv2.circle(image, (r_knee_x, r_knee_y), 7, (0, 255, 0), -1)
                    cv2.circle(image, (r_ankle_x, r_ankle_y), 7, (203, 192, 255), -1)

                    # Draw lines for visualization
                    cv2.line(image, (r_shldr_x, r_shldr_y), (r_knee_x, r_knee_y), color, 4)
                    cv2.line(image, (r_knee_x, r_knee_y), (r_ankle_x, r_ankle_y), color, 4)
                except Exception as e:
                    st.error(f"Error processing keypoints: {e}")

        #SQUATS
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
                        cur_test = "Squat"
                        cur_threshold = True
                        color = (0, 255, 0)
                        posture_text = "Correct Squat Position"
                        done_with_exercise = True

                    else:
                        posture_text = "Incorrect Squat Position"
                        color = (0, 0, 255)
                        cur_test = "Squat"
                        cur_threshold = False
                        done_with_exercise = False

                    # Display feedback on frame
                    cv2.putText(image, posture_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    cv2.putText(image, f"Knee Angle: {int(knee_angle)} | Torso Angle: {int(torso_angle)}",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    if lm.landmark[lmPose.LEFT_SHOULDER].visibility > lm.landmark[lmPose.RIGHT_SHOULDER].visibility:
                        # Mark key points
                        cv2.circle(image, (l_hip_x, l_hip_y), 7, (0, 255, 255), -1)
                        cv2.circle(image, (l_knee_x, l_knee_y), 7, (0, 255, 0), -1)
                        cv2.circle(image, (l_ankle_x, l_ankle_y), 7, (255, 0, 0), -1)
                        cv2.circle(image, (l_shldr_x, l_shldr_y), 7, (255, 255, 0), -1)

                         # Draw lines for visualization
                        cv2.line(image, (l_shldr_x, l_shldr_y), (l_hip_x, l_hip_y), color, 2)
                        cv2.line(image, (l_hip_x, l_hip_y), (l_knee_x, l_knee_y), color, 2)
                        cv2.line(image, (l_knee_x, l_knee_y), (l_ankle_x, l_ankle_y), color, 2)
                    else:
                        # Mark key points
                        cv2.circle(image, (r_hip_x, r_hip_y), 7, (0, 255, 255), -1)
                        cv2.circle(image, (r_knee_x, r_knee_y), 7, (0, 255, 0), -1)
                        cv2.circle(image, (r_ankle_x, r_ankle_y), 7, (255, 0, 0), -1)
                        cv2.circle(image, (r_shldr_x, r_shldr_y), 7, (255, 255, 0), -1)

                        # Draw lines for visualization
                        cv2.line(image, (r_shldr_x, r_shldr_y), (r_hip_x, r_hip_y), color, 2)
                        cv2.line(image, (r_hip_x, r_hip_y), (r_knee_x, r_knee_y), color, 2)
                        cv2.line(image, (r_knee_x, r_knee_y), (r_ankle_x, r_ankle_y), color, 2)

                except Exception as e:  
                    st.error(f"Error processing keypoints: {e}")

            if selected_option in ["Left Arm T", "Right Arm T", "Left Arm I", "Right Arm I", "Left Arm Y", "Right Arm Y"]:
                process_arm_position(selected_option, lm, lmPose, w, h, image)
        frame_window.image(image, channels="BGR")

    cap.release()
else:
    exercises.append(selected_option)
    done_tracker.append(done_with_exercise)
    length = len(exercises)
    if st.button("Generate Report"):
        for i in range(length):
            st.write(f"{i}: Exercise: {exercises[i]}, Done: {done_tracker[i]}")