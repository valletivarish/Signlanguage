Sign Language Recognition 
* Overview
    * This project aims to develop a sign language recognition system using computer vision techniques. The system captures hand gestures through a webcam, processes the images to extract hand landmarks, and uses a machine learning model to recognize the corresponding sign language phrases.

* Components
    1. Data Collection (collect.py)
       * Purpose: Collects sign language data from the webcam for various classes.
       * How to use:
           * Run the script (python collect.py).
           * Press 'Q' to start collecting data for each sign language class.
           * The script captures and saves images to the specified directory (DATA_DIR).


    2. Data Processing and Dataset Creation (create.py)
        * Purpose: Processes the collected images, extracts hand landmarks using MediaPipe, and creates a dataset file (data.pickle).
        * How to use:
            * Ensure DATA_DIR in create.py points to the collected data directory.
            * Run the script (python create.py).
    3. Model Training (training.py)
        * Purpose: Trains a random forest classifier on the created dataset.
        * How to use:
          * Run the script (python training.py).
          * Adjust desired_length for sequence length as needed.
    4. Real-time Testing (testing.py)
        * Purpose: Tests the trained model in real-time using the webcam.
        * How to use:
            * Run the script (python testing.py).
            * Press 'Esc' to exit the testing loop.
* Requirements
    * Python 3.x
    * OpenCV
    * MediaPipe
    * Matplotlib
    * scikit-learn
    * NumPy
* Install dependencies: pip install -r requirements.txt.
* Follow the steps in each script (collect, create, training, testing) to collect data, create a dataset, train the model, and test real-time recognition.
* Feel free to contribute, report issues, or suggest improvements!
