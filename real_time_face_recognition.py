import face_recognition
import cv2
import pickle

def load_encodings(encoding_file):
    """
    Load pre-saved encodings from a file.

    Args:
        encoding_file (str): Path to the encodings file.
    Returns:
        tuple: Known face encodings and their names.
    """
    with open(encoding_file, "rb") as f:
        data = pickle.load(f)
    return data["encodings"], data["names"]

def recognize_faces_in_frame(frame, known_encodings, known_names, tolerance=0.6):
    """
    Detect and recognize faces in a single video frame.

    Args:
        frame (ndarray): A single video frame.
        known_encodings (list): List of known face encodings.
        known_names (list): List of names corresponding to the encodings.
        tolerance (float): Distance threshold for face matching.

    Returns:
        tuple: Face locations and recognized names.
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    recognized_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=tolerance)
        distances = face_recognition.face_distance(known_encodings, face_encoding)

        if any(matches):
            best_match_index = distances.argmin()
            recognized_names.append(known_names[best_match_index])
        else:
            recognized_names.append("Unknown")

    return face_locations, recognized_names

def main(encoding_file, output_video_file):
    """
    Main function to run real-time face recognition with video recording.

    Args:
        encoding_file (str): Path to the face encodings file.
        output_video_file (str): Path to save the recorded video.
    """
    print("Loading face encodings...")
    known_encodings, known_names = load_encodings(encoding_file)

    print("Starting webcam...")
    video_capture = cv2.VideoCapture(0)  # Use default webcam (index 0)

    # Get video properties
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))

    # Video writer for recording
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for AVI files
    out = cv2.VideoWriter(output_video_file, fourcc, fps, (frame_width, frame_height))

    print("Press 'q' to quit...")

    try:
        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Failed to capture frame. Exiting...")
                break

            # Detect and recognize faces
            face_locations, face_names = recognize_faces_in_frame(frame, known_encodings, known_names)

            # Annotate the frame with rectangles and names
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Draw a rectangle around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                # Add the name label
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            # Display the annotated frame
            cv2.imshow("Real-Time Face Recognition", frame)

            # Write the frame to the output video
            out.write(frame)

            # Break the loop on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Release resources
        video_capture.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Video saved to {output_video_file}")

if __name__ == "__main__":
    encoding_file = "face_encodings.pkl"  # Path to your encodings file
    output_video_file = "output.avi"      # Path to save the recorded video
    main(encoding_file, output_video_file)
