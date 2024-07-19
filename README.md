# Artificial Vision Project: Age Estimation from faces (PyTorch)

<div align='center'>
  <img src="AgeEstimation.png" alt="Age Estimation" style="width:50%; max-width:400px;">
</div>

## Overview
Estimating the age from faces is a very challenging task as there are numerous aspects that can influence this choice such as ethnicity or gender. Moreover, the faces of people of the same age may have a very different appearance due to the characteristics of the person represented because of the uncontrolled ageing process or due to image’s characteristics such as pose, lighting or quality. Likewise automatic age estimation from face images has numerous practical
applications such as demographic statistic collection, customer profiling, assistance of biometrics, etc.

This system to estimate the age from faces has been developed in the context of a competition, [the Guess The Age (GTA) contest](https://gta2021.unisa.it/),  in which it has been made available the MIVIA Age Dataset, a very large Dataset composed of 575.073 images of more than 9.000 identities of different ages, very heterogeneous in terms of face size, illumination conditions, facial pose, gender and ethnicity. The dataset distribution is not homogeneous for the different ages and there are very
few samples for the external ages, reproducing a tailed distribution.

## Developed System
<div align="center">
  <img src="/utils/Complete_architecture.png" alt="Architecture of the developed system.">
</div>

The developed system has been designed being aware of the complexity of the task to be carried out and of the problems that would arise from the distribution of the dataset, with the aim of realising a performing product that would not require excessive
resources and above all that would guarantee the explainability property of the architecture. For all these reasons, the designed architecture is made of different **weak learners**,
each one with a specific task, consequently with a specific architecture and trained in different times, using as features extractor the InceptionResnetV1, starting from already trained weights and making fine-tuning on it. The **ensemble of the weak learners** is all about to obtain a stronger learner with more varied functionality. The problem was faced with a regression approach rather than classification one, precisely to deal with classes' imbalance, even if regression problems are generally more complex to handle than classification ones due to the continuous value of the outputs.
The general architecture of the proposed model is a **hybrid architecture**, composed of an initial backbone (InceptionResnetV1 pretrained on VGGFace2 dataset), two classifiers and a regressor. Some of the backbone layers have been cloned in order to obtain two heads, one tied with the two classifiers and the other tied with the regressor. 
- **The two classifiers** have the task to give support to the regression task reducing the range within wich to predict the continuous age value. In particular they have to predict the age label of a person, based on 81 classes, corresponding to the possible ages allowed by our system, avoiding the errors due to "edge samples". Both classifiers have been trained considering the output as the argmax on the Softmax’s resultant vector, but this output was processed, before giving it to regressor, obtaining a range of ages containing the predicted one. In particular, starting from the output of each classifier, through SoftMax, are obtained the probabilities that a sample belongs to each class, then is computed the argmax on these probabilities and this corresponds to classifier’s output. Then the age range predicted by each classifier is the range of four years before and after the age corresponding to the argmax above mentioned. Then for each classifier is created a vector of 81 values, all zeroes except for the nine (4+4+1) indices corresponding to the obtained range. The vectors of each classifier are concatenated, first that of the First Classifier and then that of the Second
Classifier, obtaining a total vector of 162 values.
What is different between the two classifiers is the applied training procedure; in fact, for both it was used a boosting procedure employing the entire dataset, without employing any bagging procedure. So, the only difference is that, during the training procedure, the First Classifier considers all the samples with the same weights equal to one, while the Second Classifier considers with higher weights the samples misclassified by the first one and with lower weights the samples correctly classified by the first one. Obviously, during the training procedure, it was taken into account the inequality of errors depending on the number of years incorrectly predicted.

- **The Regressor** is composed of two different parts. The first one aims to extract features that are useful for the regression task. These features, concatenated with age ranges prediction coming from classifiers’ outputs combination are given as input to the second regressor's part, in order to obtain the final age's prediction.

## Training and Testing procedure
The dataset was randomly devided in training and validation set (74%-26%) respecting the distribution of the entire dataset. According to contest rule, AAR (Age Accuracy and Regularity) is used to evaluate the performance of the model.
Face aligment technique is employed during the pre-processing phaese. Also during the training phase, different augmentation techniques have been employed, like flipping, rotation and brightness variation

To train the Classifiers it was used the Categorical Cross Entropy using custom weights, computed also using the LDS (Label Distribution Smoothing technique), to take into account the imbalance of the dataset. 
To train the Regressor a **custom loss function** given by the sum of sigma and mMAE was invented.
For more details about the logic behind the losses used and designed, the training procedure performed, the designed architecture and the obtained results, refer to the [Model Architecture Documentation](2023_GTA_Contest_Report_Template_Definitivo.pdf).

## Repository Structure
- `modelAndScripts/`: contains all the codes used for single experts and total model training, validation and architecture definition;
- `utils`: contains some image used only to test that the test code works correctly (the test set is not available before system submission), the obtained results, the weights of the model used for face alignment and the weights of the final developed system for age estimation;
- `2023_GTA_Contest_Report_Template_Definitivo.pdf`: is the final report containing descriptive diagrams of the realised architecture, the motivations underlying certain design choices, details on the training procedures of individual experts and of the overall model, details on the loss functions employed and developed and obtained results;
- `test.ipynb`, `test.py` and `testFast.py`: contains the code to test, on some examples images, that everything works well and so that the developed model is correctly upload and that it is possible to run the inference.

## Feedback
For any feedback, questions or inquiries, please contact the project maintainers listed in the Contributors section.
