This project focuses on solving the traffic sign classification challenge using a CNN implemented in TensorFlow, achieving an impressive 99.33% accuracy. Key aspects of this solution include data preprocessing, augmentation, pre-training, and skip connections in the network architecture. The dataset used is the German Traffic Sign Dataset, which is part of Udacity's Self-Driving Car Nanodegree program, but is also publicly available here.

Dataset
The German Traffic Sign Dataset consists of 39,209 color images (32×32 pixels) for training and 12,630 images for testing. Each image belongs to one of 43 traffic sign classes. The pixel values range from [0, 255] in the RGB color space, and each class is labeled as an integer between 0 and 42. The dataset is imbalanced, with some classes having far fewer examples than others. Additionally, the images exhibit variability in contrast and brightness, which calls for the application of techniques like histogram equalization to enhance feature extraction.

Preprocessing
Preprocessing typically involves scaling the pixel values to the [0, 1] range (since they are originally in [0, 255]), one-hot encoding the labels, and shuffling the dataset. Due to the variability in the dataset, localized histogram equalization is also applied to improve feature extraction. The model is designed to work with grayscale images rather than full-color images, as previous research by Pierre Sermanet and Yann LeCun suggests that color channels don't significantly enhance performance. Therefore, the Y channel from the YCbCr color space is used.

Data Augmentation
The dataset is relatively small and unbalanced, making it difficult for the model to generalize well. To address this, data augmentation is applied to increase the dataset size and balance the class distribution.

Flipping
Some traffic signs are invariant to horizontal or vertical flipping, meaning that flipping an image of these signs will not change their class. For example, signs like Priority Road and No Entry can be flipped without changing their meaning, while others require 180-degree rotation (achieved by flipping both horizontally and vertically). This method increases the training examples from 39,209 to 63,538 at no additional data collection cost.

Rotation and Projection
Further augmentation is done using random rotations and projections. Other transformations, such as blur, noise, and gamma adjustments, were tested, but rotation and projection gave the best results. Projection also handles random shearing and scaling by altering the image corners within a specified range.

Model Architecture
The model is a deep neural network based on a CNN architecture inspired by Daniel Nouri's tutorial and the Sermanet/LeCun paper. The model consists of 3 convolutional layers for feature extraction and 1 fully connected layer for classification.

<p align="center"> <img src="model_architecture.png" alt="Model architecture"/> </p>
Instead of following a strict feed-forward architecture, this network employs multi-scale features. This means that the output from the convolutional layers is not only passed to the next layer but also fed directly into the fully connected layer, after additional max-pooling, ensuring that all convolutions are appropriately subsampled before classification.

Regularization
Several regularization techniques were used to minimize overfitting:

Dropout: Applied to both convolutional and fully connected layers to improve generalization. Though convolutional layers are naturally good regularizers, small amounts of dropout were found to slightly improve performance.
graphql
Copy code
                Type           Size         keep_p      Dropout
 Layer 1        5x5 Conv       32           0.9         10% of neurons  
 Layer 2        5x5 Conv       64           0.8         20% of neurons
 Layer 3        5x5 Conv       128          0.7         30% of neurons
 Layer 4        FC             1024         0.5         50% of neurons
L2 Regularization: Applied with a lambda value of 0.0001, L2 regularization is used for the weights in the fully connected layers to reduce overfitting without affecting the bias terms.

Early Stopping: Early stopping is implemented with a patience of 100 epochs. The model is trained until the validation loss no longer improves, at which point the best-performing weights are retained.

Training
Two datasets were used for training:

Extended Dataset: Augmented to contain 20 times the data of the original dataset, with each image generating 19 additional versions through jittering, resulting in improved model performance.

Balanced Dataset: Adjusted to include 20,000 examples per class, balancing the dataset. Each class is augmented with jittered images to reach 20,000 samples.

Training Stages
Stage 1: Pre-training. The model is pre-trained using the extended dataset with a learning rate of 0.001. This stage typically converges after about 180 epochs (~3.5 hours on an Nvidia GTX1080 GPU).

Stage 2: Fine-tuning. The model is fine-tuned on the balanced dataset with a reduced learning rate of 0.0001. Fine-tuning further boosts test set accuracy.

This two-stage training process can easily achieve over 99% accuracy on the test set.

Results
After multiple rounds of fine-tuning, the model achieved 99.33% accuracy on the test set. Given that the test set contains 12,630 images, this means the model misclassified 85 images. Most errors involved images with artifacts, such as shadows or obstructions, or classes underrepresented in the training data. Improving color information or training solely on balanced datasets could mitigate these issues.

Human performance on similar tasks ranges between 98.3% and 98.8%, meaning this model exceeds average human accuracy, demonstrating the effectiveness of machine learning in this domain.
