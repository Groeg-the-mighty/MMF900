clear all

% ----- Change these ------------------------------------------------------
K           = 5;
Directory   = "XXXX";  % Change to correct local directory
Ydata_train = append(Directory, "\Ydata_train_example_case.csv");
Ydata_test  = append(Directory, "\Ydata_test_example_case.csv");
Xdata_train = append(Directory, "\Xdata_train_example_case.csv");
Xdata_test  = append(Directory, "\Xdata_test_example_case.csv");
plot_if     = 0;                                                % 1 for plot, 0 for no plot
%--------------------------------------------------------------------------

%%
fprintf('\n Loading data... '); tic

Xdata      = readmatrix(Xdata_train)/255;
ydata      = readmatrix(Ydata_train);
Xdata_test = readmatrix(Xdata_test)/255;
ydata_test = readmatrix(Ydata_test);
n = length(ydata_test);
clearvars Xdata_all ydata_all
fprintf('Data splitted \n'); toc 

fprintf('\n Adjusting data... '); %tic
xmean          = mean(Xdata, 1); % mean pixel value (lengthy computation time)
Xdata_adj      = (Xdata - xmean);
Xdata_test_adj = (Xdata_test - xmean);
clearvars Xdata Xdata_test
fprintf('Data adjusted \n'); toc 

%%
fprintf('\n Calculating covariance matrix... '); %tic
C = COVARIANCE(Xdata_adj);
fprintf('Covariance matrix calculated \n'); toc 

%%
fprintf('\n Calculating SVD... '); %tic
[U, S, V] = svds(C,K);
fprintf('SVD calculated \n'); toc
clearvars C


fprintf('\n Calculating PCA matices... '); %tic
PCA = Xdata_adj*V; % = U.*S; % needs implimenting
PCA_test = Xdata_test_adj*V;
fprintf('PCA matrices calculated \n'); toc 
clearvars Xdata_adj Xdata_test_adj U S V 

%%
n_correct     = 0;
y_pred_data   = zeros(n,1);
wrong_label   = zeros(n,2);      % stores index and guesses of wrong guesses
right_guesses = zeros(2,1);     
wrong_guesses = zeros(2,1);
all_guesses   = zeros(2,1);      % counts all digits
fprintf('\n Starting prediction procedure... '); %tic
for i=1:n
    d=vecnorm(PCA_test(i,:)-PCA,2,2);
    [~, number] = min(d); % returns smallest element

    if ydata_test(i)==ydata(number)
        n_correct = n_correct+1;
        right_guesses(ydata(number)+1) = right_guesses(ydata(number)+1) + 1;
    else
        wrong_guesses(ydata(number)+1) = wrong_guesses(ydata(number)+1) + 1;
        wrong_label(i-n_correct,1) = i + length(ydata);
        wrong_label(i-n_correct,2) = ydata(number);
    end
    all_guesses(ydata(number)+1) = all_guesses(ydata(number)+1) + 1;
    y_pred_data(i) = ydata(number);
end

precision(1) = right_guesses(1)/(right_guesses(1)+wrong_guesses(2));
precision(2) = right_guesses(2)/(right_guesses(2)+wrong_guesses(1));

wrong_label(wrong_label(:,1) == 0,:) = [];  % removes unwanted length of array
Pred_rate =  sum(right_guesses)/length(ydata_test);
E         =  sum(wrong_guesses)/sum(all_guesses);
fprintf('Finished prediction procedure! \n'); toc 
