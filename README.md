# Celebrity-image-classification

This project explains in detail a Python code created for classifying images using Convolutional Neural Networks (CNNs). The goal is to classify pictures of celebrities like Lionel Messi, Maria Sharapova, Roger Federer, Serena Williams and Virat Kohli into different categories.
1.	Data Preprocessing:
The process starts by loading and preparing the image data. Using the OpenCV library, the images are read, transformed into RGB format with the Pillow library, and resized to a consistent 128 * 128 pixel dimension. Following this, the dataset is created by combining the images and associating them with integer labels representing various celebrities, ranging from 0 to 4.
2.	Train-Test Split:
The dataset is divided into training and testing sets using the 'train_test_split' function from the scikit-learn library. This ensures a suitable separation of data for both training the model and assessing its performance.
3.	Data Normalization:
Pixel values of the images in both the training and testing sets are normalized by dividing the floating-point values of both by 255. This step is crucial for standardizing the input data and facilitating the training process.
4.	Convolutional Neural Network Architecture:
The Convolutional Neural Network (CNN) model is build using TensorFlow's Keras module with the Sequential API. The structure includes three convolutional layers using Rectified Linear Unit (ReLU) activation, followed by corresponding max-pooling layers. Afterward, the output is flattened, and two dense layers with ReLU activation are introduced. The ultimate dense layer consists of five units with softmax activation, signifying the five classes.
5.	Model Compilation and Training:
The model is configured by employing the Adam optimizer and sparse categorical cross-entropy loss during compilation. To avoid overfitting, an early stopping callback is integrated, which monitors accuracy and halts training if there is no improvement within a patience period of 10 epochs. The training is conducted across 20 epochs, and the model's performance is observed throughout this process.
6.	Evaluation and Results:
Following the training phase, the model undergoes evaluation on the test set, and the resulting accuracy is displayed. Moreover, predictions are generated for a sample image using the 'make_prediction' function.
7.	Conclusion and Recommendations
In summary, the image classification script effectively utilizes a Convolutional Neural Network to identify celebrities in images, showcasing impressive accuracy on the test set. The integration of early stopping enhances training efficiency and mitigates overfitting. For future improvements, one could explore alternative architectures, fine-tune hyperparameters, and ensure a balanced distribution of images across different classes to address potential biases. Regular updates and retraining with new data would allow the model to adapt to changing patterns and sustain optimal accuracy.



