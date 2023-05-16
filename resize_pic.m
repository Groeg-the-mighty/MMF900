c = 'resize_';
folder = "C:\Users\lulle\Documents\MMF900\ISIC_2020_Training_JPEG\ISIC_2020_Training_Input\";
% C:\Users\lulle\Documents\MMF900\ISIC_2020_Training_JPEG\ISIC_2020_Test_Input\
folder_save = "C:\Users\lulle\Documents\MMF900\ISIC_2020_Training_JPEG\ISIC_2020_Training_Input_Resized\";
% C:\Users\lulle\Documents\MMF900\ISIC_2020_Training_JPEG\ISIC_2020_Test_Input_Resized\

filePattern = fullfile(folder, '*.jpg');
theFiles = dir(filePattern);
rows = 100; 
cols = 150; 
n = length(theFiles) % 1000; %length(theFiles)

tic
for k = 1 : n
    
    pic = imread(fullfile(folder, theFiles(k).name));
    filename = append(c , theFiles(k).name);
    [x,y,z] = size(pic);
    
    new_pic = imresize(pic, [rows,cols]); 
    new_pic = cast(new_pic, 'uint8');
    
    imwrite(new_pic, append(folder_save,filename));
end
toc