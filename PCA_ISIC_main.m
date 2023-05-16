

% ----- Change these ------------------------------------------------------
K           = 1000;
n           = 5000;     % number of images to test
y_directory = "C:\Users\merila\Desktop\MMF900\training_short.csv"; % pictures.csv
X_directoy  = "C:\Users\merila\Desktop\MMF900\2020_with_2019malignant"; % cancer.csv
plot_if     = 0;                                                % 1 for plot, 0 for no plot
%--------------------------------------------------------------------------

%% Time estimation
if K == 10
    %Tsvd = 82;
    Tpca = 150; 
elseif K == 100
    %Tsvd = 805;
elseif K == 1000
    %Tsvd = 8864; 
    Tpca = 1321;
else
    %Tsvd = 0;
    Tpca = 0;
end
T = [252, 4, 6, 507, 9*K, 77, Tpca]; % computer specific values 
fprintf('\n Estimated completion time for whole program: %s ', datetime + seconds(sum(T)));

%%
fprintf(['\n Loading data... ' ...
    '\n Estimated completion time: %s '], datetimet + seconds(T(1))); tic
Xdata_all  = readmatrix(X_directoy)/255;  % normalized to reduce computational loss
ydata_all  = readmatrix(y_directory);
fprintf('Data loaded \n'); toc % 252 seconds

fprintf(['\n Spliting data... ' ...
    '\n Estimated completion time: %s '], datetime + seconds(T(2))); tic
Xdata      = Xdata_all(1:end-n, :);
ydata      = ydata_all(1:end-n, 2);
Xdata_test = Xdata_all(end-n +1: end, :);
ydata_test = ydata_all(end-n +1: end, 2);
clearvars Xdata_all ydata_all
fprintf('Data splitted \n'); toc % 4 seconds

fprintf(['\n Adjusting data... ' ...
    '\n Estimated completion time: %s '], datetime + seconds(T(3))); tic
xmean          = mean(Xdata, 1); % mean pixel value (lengthy computation time)
Xdata_adj      = (Xdata - xmean);
Xdata_test_adj = (Xdata_test - xmean);
clearvars Xdata Xdata_test
fprintf('Data adjusted \n'); toc % 6 seconds
% data could be adjusted before splitting

fprintf(['\n Calculating covariance matrix... ' ...
    '\n Estimated completion time: %s '], datetime + seconds(T(4))); tic
C = COVARIANCE(Xdata_adj);
fprintf('Covariance matrix calculated \n'); toc % 507 seconds

%%
fprintf(['\n Calculating SVD... ' ...
    '\n Estimated completion time: %s '], datetime + seconds(T(5))); tic
[U, S, V] = svds(C,K);
fprintf('SVD calculated \n'); toc
%clearvars C
% times
% k = 1     and C = 45000x45000 takes       seconds
% k = 10    and C = 45000x45000 takes    82 seconds
% k = 20    and C = 45000x45000 takes       seconds 
% k = 100   and C = 45000x45000 takes   805 seconds 
% k = 200   and C = 45000x45000 takes       seconds
% k = 1000  and C = 45000x45000 takes  8864 seconds
% k = 10000 and C = 45000x45000 takes       seconds

fprintf(['\n Calculating PCA matices... ' ...
    '\n Estimated completion time: %s '], datetime + seconds(T(6))); tic
PCA = Xdata_adj*V; % = U.*S; % needs implimenting
PCA_test = Xdata_test_adj*V;
fprintf('PCA matrices calculated \n'); toc % 77 seconds(k=10)

%%
n_correct     = 0;
y_pred_data   = zeros(n,1);
wrong_label   = zeros(n,2);      % stores index and guesses of wrong guesses
precision     = zeros(2,1);
something     = zeros(2,1);
all_guesses   = zeros(2,1);      % counts all digits
fprintf(['\n Starting prediction procedure... ' ...
    '\n Estimated completion time: %s '], datetime + seconds(T(7))); tic
for i=1:n
    min_d=100000;
    for j=1:length(ydata)
        d=norm(PCA_test(i,:)-PCA(j,:));
        if d<min_d
            min_d=d;
            number=j;
            % y_pred_datatest(j) = ydata(number);
        end
    end
    if ydata_test(i)==ydata(number)
        n_correct = n_correct+1;
        precision(ydata(number)+1) = precision(ydata(number)+1) + 1;
    else
        something(ydata(number)+1) = something(ydata(number)+1) + 1;
        wrong_label(i-n_correct,1) = i + length(ydata);
        wrong_label(i-n_correct,2) = ydata(number);
    end
    all_guesses(ydata(number) +1) = all_guesses(ydata(number) +1) + 1;
    y_pred_data(i) = ydata(number);
end

% Better way to do the same?
precision(1) = precision(1)/(precision(1)+something(2));
precision(2) = precision(2)/(precision(2)+something(1));

wrong_label(wrong_label(:,1) == 0,:) = [];
Pred_rate =  n_correct/length(ydata_test);
E = length(wrong_label(:,1))/sum(all_guesses);
fprintf('Finished prediction procedure! \n'); toc 
% times
% k = 1     and C = 45000x45000 takes       seconds
% k = 10    and C = 45000x45000 takes   150 seconds
% k = 20    and C = 45000x45000 takes       seconds 
% k = 100   and C = 45000x45000 takes       seconds 
% k = 200   and C = 45000x45000 takes       seconds
% k = 1000  and C = 45000x45000 takes  1321 seconds
% k = 10000 and C = 45000x45000 takes       seconds
[a,Fs] = audioread('rumble.mp3');
soundsc(a,Fs)

confusionchart(ydata_test, y_pred_data);
confusionmat(ydata_test, y_pred_data)

% F(1) = load('gong');
% F(2) = load('chirp');
% F(3) = load('train');
% sound(F(1).y,F(1).Fs)
% sound(F(2).y,F(2).Fs)
% sound(F(3).y,F(3).Fs)



