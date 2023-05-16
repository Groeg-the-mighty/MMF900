
folder_pics = "C:\Users\lulle\Documents\MMF900\ISIC_2020_Training_JPEG\ISIC_2020_Training_Input_Resized\";
% "C:\Users\lulle\Documents\MMF900\ISIC_2020_Test_JPEG\ISIC_2020_Test_Input_Resized";        % Path : Folder with all training pictures
save_to = "C:\Users\lulle\Documents\MMF900\training_pictures_python.csv";                    % Path : Name of document for new shorter metadata                                                                           
% "C:\Users\lulle\Documents\MMF900\testing_pictures.csv";

filePattern = fullfile(folder_pics, '*.jpg');
theFiles = dir(filePattern);
n =  length(theFiles); %14500 %1000;     % Number of pics
rows = 100;
cols = 150; 
size = rows*cols*3 % The dimension of the imported picture. Check the dimencion in resize_pic 
arr = uint8( zeros( n + 1 , rows*cols*3) );

tic
for k = 2 : n + 1 % OBS måske det skal være K = 2 : n+1 for at få 0 på første række  
    pic = imread(fullfile(folder_pics, theFiles(k-1).name));
    new_pic_red = reshape(pic(:,:,1), 1, rows*cols)  ;
    new_pic_green = reshape(pic(:,:,2), 1, rows*cols) ;
    new_pic_blue = reshape(pic(:,:,3), 1, rows*cols) ;
    arr(k,:) = [new_pic_red, new_pic_green, new_pic_blue]; 
end


writematrix(arr, save_to);
toc
