## MMF900
Analysis of ISIC and MNIST datasets using Matlab and python scripts. 


## READ ME; 

# PCA: MNIST dataset

- Download datasets 'mnist_test.csv' and 'mnist_training.csv' using the link http://makeyourownneuralnetwork.blogspot.com/2015/03/the-mnist-dataset-of-handwitten-digits.html as datasets are to large for us to 
  upload.  
- Download matlab code 'PCA_MNIST_github.m'
- Change directory to local downloaded files  
- Run the code
- Output: Missclassification rate and vector with prediction rate for each digits

# CNN: MNIST dataset  

- Download 'CNN_MNIST.py'
- MNIST dataset is available through python and is imported directly through the code
- Run the code
- Returns: Missclassification, accuracy vs epochs and a confusion matrix

# PCA: ISIC dataset 

- Download the four dataset samples: 'Xdata_test_example_case.csv', 'Xdata_train_example_case.csv', 'Ydata_test_example_case.csv', 'Ydata_train_example_case.csv'
- Download matlab files: 'COVARIANCE.m' and 'PCA_ISIC_main.m'
- Run the code
- Returns: Missclassification, Confusion matrix, prediction rate  

# CNN: ISIC dataset 

- Download the four dataset sample: 'Xdata_test_example_case.csv', 'Xdata_train_example_case.csv', 'Ydata_test_example_case.csv', 'Ydata_train_example_case.csv'
- OBS: Python swallows the first line in each csv file, hence the testing and training will only be conducted on 99 pictures
- Download python file 'CNN_ISIC_github.py'
- Run the code
- Returns: Missclassification, confusion matrix and accuracy vs. epochs  
