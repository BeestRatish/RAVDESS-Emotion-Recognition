import numpy as np
from sklearn.preprocessing import LabelEncoder

def generate_label_classes():
    # Define the emotions in the same order as used during training
    emotions = [
        'neutral',
        'calm',
        'happy',
        'sad',
        'angry',
        'fearful',
        'disgust',
        'surprised'
    ]
    
    # Create and fit the label encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(emotions)
    
    # Save the classes
    np.save('models/label_classes.npy', label_encoder.classes_)
    print("Label classes saved successfully!")
    print("Classes:", label_encoder.classes_)

if __name__ == "__main__":
    generate_label_classes() 