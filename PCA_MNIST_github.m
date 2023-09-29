%% Prediction rate

clear
close all
clc


%% ----------- INPUT ------------
% rng(100);                           % Set seed 
number_examples_train = 60000;      % Number of training data
n_iter = 1;                         % Iteration for randomness elimination  
n = 10000;                          % Size of test data                    
k = 10                              % Dimensions to keep 


%% --------- IMPORT AND SPLITTING DATA -------------
fprintf('Reading data\n')
data_train = csvread('mnist_train.csv'); % Import training data
data_test = csvread('mnist_test.csv');   % Import test data 
fprintf('Data read\n')

fprintf(' Adjusting data\n')
Xdata = data_train(1:number_examples_train, 2:end); % Save training data
ydata = data_train(1:number_examples_train, 1)';    % Save correct digit in picture for training data

xmean = mean(Xdata, 1);                             % Mean of training data
Xdata_adj = (Xdata - xmean);                        % Adjusted for mean  
fprintf(' Data adjusted\n')

%% ------- Test data ---- 
fprintf(' Splitting data \n')
Xdata_n = data_test(1:n, 2:end); % Used for n = 10000
ydata_n = data_test(1:n, 1)';    % Used for n = 10000
Xdata_n_adj = (Xdata_n - xmean);    % Test data adjusted for mean
fprintf(' Data splitted \n')

%% ----------- COVARIANCE MATRIX -----------------
fprintf(' Calculating covariance matrix \n')
C = cov(Xdata_adj);                 % covariance matrix for adjusted training data
fprintf(' Covariance matrix calculated \n')

%% ----------- SVD ----------------
fprintf(' Calculating SVD decomposition\n')
[U,S,V] = svd(C);                   % Singular value decomposition for covariance matrix
fprintf(' SVD decomposed \n')

fprintf(' Calculating PCA matrices \n')
PCA = Xdata_adj * V(:,1:k);         % PCA training
PCA_n = Xdata_n_adj * V(:,1:k);     % PCA testing 
fprintf(' PCA matrices calculated\n')

%% -----------  RUN  -------------
 
plotting=1;                         % Plot if true  

n_correct = 0;
y_pred_data = zeros(n,1);
corr_digit = zeros(10,1);      % counts correct digits
false_digit = zeros(10,1);     % counts missclassified digits
all_digit = zeros(10,1);       % counts all digits

fprintf(' Starting prediction procedure\n')
tic


for i=1:n                      % test data size
    min_d = 100000;
    for j = 1:length(ydata)    % training data size
        d = norm(PCA_n(i,:) - PCA(j,:));
        if d < min_d
            min_d = d;
            number = j;
        end
    end
    if ydata_n(i) == ydata(number)
        n_correct = n_correct+1;
        corr_digit(ydata(number) + 1)  = corr_digit(ydata(number) + 1) + 1;
    else
        false_digit(ydata(number) + 1) = false_digit(ydata(number) + 1) + 1;
    end
    all_digit(ydata(number) + 1) = all_digit(ydata(number) + 1) + 1;
    y_pred_data(i) = ydata(number);
end
E = sum(false_digit) / sum(all_digit)    % Miss-classification rate
prediction = corr_digit./all_digit   % Precision for the different classes

toc
fprintf(' Prediction procedure done\n')
