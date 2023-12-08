# Celebrity-image-classification

This project provides a comprehensive overview of the Python script developed for image classification using Convolutional Neural Networks (CNNs). It is designed for a multi-class classification task involving images of celebrities, including Lionel Messi, Maria Sharapova, Roger Federer, Serena Williams, and Virat Kohli.
Data Preprocessing
It begins with the loading and preprocessing of image data. Images are read using the OpenCV library, converted to RGB format using the Pillow library, and resized to a standardized dimension of 128 * 128 pixels. The dataset is then constructed, with images appended along with corresponding integer labels representing different celebrities (0 to 4).
Train-Test Split
The dataset is split into training and testing sets using the ‘train_test_split’ function from the scikit-learn library. This ensures an appropriate division of data for model training and evaluation.
Data Normalization
Pixel values of the images in both the training and testing sets are normalized by dividing the floating point values of both by 255. This step is crucial for standardizing the input data and facilitating the training process.
Convolutional Neural Network Architecture
The CNN model is constructed using the Sequential API from TensorFlow's Keras module. The architecture comprises three convolutional layers with Rectified Linear Unit (ReLU) activation, each followed by max-pooling layers. Subsequently, the output is flattened, and two dense layers with ReLU activation are added. The final dense layer has five units with softmax activation, representing the five classes.
Model Compilation and Training
The model is compiled using the Adam optimizer and sparse categorical cross-entropy loss. To prevent overfitting, an early stopping callback is implemented, monitoring accuracy with a patience of 10 epochs. The training process is executed over 20 epochs, and the model's performance is monitored.

Evaluation and Results
After training, the model is evaluated on the test set, and the accuracy is printed. Additionally, predictions are made on a sample image using the ‘make_prediction’ function.
Conclusion and Recommendations
In conclusion, the image classification script successfully deploys a Convolutional Neural Network for recognizing celebrities from images, demonstrating commendable accuracy on the test set. The incorporation of early stopping contributes to efficient training and prevents overfitting. To further enhance model performance, future iterations may explore alternative architectures, fine-tune hyperparameters, and ensure a balanced distribution of images among different classes to mitigate biases. Regular updates and retraining with new data can help the model adapt to evolving patterns and maintain optimal accuracy.

