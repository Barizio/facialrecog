import face_recognition
import os
import pickle
from PIL import Image

def encode_faces(dataset_path, output_file):
    """
    Encodes faces from the dataset and saves them to a file.

    Args:
        dataset_path (str): Path to the dataset directory.
        output_file (str): Path to save the encoding file.
    """
    known_encodings = []
    known_names = []

    # Loop through each person's folder in the dataset
    for person_name in os.listdir(dataset_path):
        person_folder = os.path.join(dataset_path, person_name)

        # Skip non-directory files
        if not os.path.isdir(person_folder):
            continue

        print(f"Processing images for: {person_name}")

        # Loop through each image in the person's folder
        for image_name in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_name)
            
            try:
                # Load the image
                image = face_recognition.load_image_file(image_path)

                # Get the face encodings (assume one face per image)
                encodings = face_recognition.face_encodings(image)

                if len(encodings) > 0:
                    known_encodings.append(encodings[0])
                    known_names.append(person_name)
                else:
                    print(f"No face detected in {image_name}. Skipping...")
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
    
    # Save encodings and names to a file
    data = {"encodings": known_encodings, "names": known_names}
    with open(output_file, "wb") as f:
        pickle.dump(data, f)

    print(f"Encodings saved to {output_file}")

if __name__ == "__main__":
    # Path to the dataset and output file
    dataset_path = "dataset"  # Folder containing subfolders with images of each person
    output_file = "face_encodings.pkl"  # File to save the encodings
    encode_faces(dataset_path, output_file)
