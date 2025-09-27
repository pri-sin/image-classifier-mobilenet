# Simple Image Classifier using MobileNetV2 📸

A straightforward command-line tool that uses a pre-trained TensorFlow model (**MobileNetV2**) to classify the main subject in a folder of images.

This project is a great starting point for anyone looking to understand and implement deep learning for image recognition. It's designed to be simple to set up and run.



## Features ✨

-   **Powerful Model**: Leverages **MobileNetV2**, a state-of-the-art model pre-trained on the massive ImageNet dataset.
-   **Easy to Use**: Simply place your images in an `images` folder and run the script.
-   **Batch Processing**: Classifies all images in the folder and prints a neat summary of the results.
-   **Confidence Score**: Shows the model's confidence level for each prediction.

---

## How It Works

The script loads the pre-trained MobileNetV2 model using TensorFlow/Keras. For each image in the `images` directory, it resizes it to the required 224x224 pixels, preprocesses it, and feeds it to the model. The model then returns a prediction, and the script displays the top guess with its confidence score.

---

## Setup & Usage 🚀

### Prerequisites

-   Python 3.8+
-   TensorFlow

### Steps

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/pri-sin/image-classifier-mobilenet.git]
    cd yimage-classifier-mobilenet
    ```

2.  **Install Dependencies:**
    It's highly recommended to use a virtual environment.
    ```bash
    # Create and activate a virtual environment (optional but recommended)
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

    # Install TensorFlow
    pip install tensorflow
    ```

3.  **Add Your Images:**
    Create a folder named `images` in the main project directory and place all the `.jpg`, `.png`, or `.jpeg` files you want to classify inside it.

    ```
    your-repository-name/
    ├── your_script_name.py
    └── images/
        ├── cat.jpg
        ├── car.png
        └── sunset.jpeg
    ```

4.  **Run the Script:**
    Execute the script from your terminal. It will automatically find and process the images.
    ```bash
    python image-classifierpy
    ```

---

## Example Output

After running, you will see output in your terminal like this:

```
cat.jpg: Egyptian_cat (84.51%)
car.png: sports_car (91.23%)
sunset.jpeg: seashore (67.89%)
```

**Note:** The model was trained on the ImageNet dataset, which contains 1,000 common object categories. It may not recognize highly specialized or abstract images.
