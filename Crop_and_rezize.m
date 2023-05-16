% This program resizes selected pictures to a given size and crops metadata
% to include only the cropped pictures.
% The input data is two sets of picturefolders with related metadata. The
% first set is sorted with only malignent cases remaining while all of the
% second set is used
% The pictures are named in the order they are handeled in order to
% preserve the picture and metadata relation.

% 7500 seconds for 33126 + <5000 pictures

clear all

folder_pics_1 = "C:\Users\kyhng\Desktop\Machine_learning_image_recognition\ISIC_2019_Training_Input"; % 2019 pictures
folder_pics_2 = "C:\Users\kyhng\Desktop\Machine_learning_image_recognition\train"; %2020 pictures
folder_info_1 = "C:\Users\kyhng\Desktop\Machine_learning_image_recognition\ISIC_2019_Training_GroundTruth.csv";
folder_info_2 = "C:\Users\kyhng\Desktop\Machine_learning_image_recognition\ISIC_2020_Training_GroundTruth.csv";
y_save_to = "C:\Users\kyhng\Desktop\Machine_learning_image_recognition\training_short.csv";
X_foldert_save_to = "C:\Users\kyhng\Desktop\Machine_learning_image_recognition\ResizedLarge\";


tic

filePattern1 = fullfile(folder_pics_1, '*.jpg'); % returns directory with general specified ending (.jpg)
theFiles1 = dir(filePattern1); % returns array of files with filepattern
size_arr1 = length(theFiles1);
info1 = readtable(folder_info_1);

filePattern2 = fullfile(folder_pics_2, '*.jpg'); % returns directory with general specified ending (.jpg)
theFiles2 = dir(filePattern2); % returns array of files with filepattern
size_arr2 = length(theFiles2);
info2 = readtable(folder_info_2);

target = zeros(size_arr1 + size_arr2,1);
name = string(target); % string array in correct dimensions

c = "resized_";
rows = 100; % choose mod 2
cols = 150; % choose mod 2, Should be the larger value
i = 1;

%% 2019 files only with malignant
for k = 1 : length(info1.MEL)
    f_name = append( info1.image{k},'.jpg');
    cell = struct2cell(theFiles1);
    exist = find(string(cell(1,:)) == string(f_name));
    if( ~isempty(exist) )
        if info1.MEL(k) == 1 % only malignant
            % -------- croping metadata ---------------
            name(i) = append(num2str(i), info1.image{k},'.jpg');
            target(i) = info1.MEL(k);

            % ----------- rezising picture -------------
            file_name = append(num2str(i), c,theFiles1(k).name);

            pic = imread(fullfile(folder_pics_1, theFiles1(k).name));
            [x,y,z] = size(pic);

            new_pic = imresize(pic, [rows, cols]);

            new_pic = cast(new_pic,'uint8');
            imwrite(new_pic, append(X_foldert_save_to, file_name))

            i = i + 1;
        end
    end
end
fprintf('2019 done \n')
%% 2020 all files

for k = 1 : length(info2.target)
    f_name = append( info2.image_name{k},'.jpg');
    cell = struct2cell(theFiles2);
    exist = find(string(cell(1,:)) == string(f_name));
    if( ~isempty(exist) )
        % -------- croping metadata ---------------
        name(i) = append(num2str(i),info2.image_name{k},'.jpg');
        target(i) = info2.target(k);

        % ----------- rezising picture -------------
        file_name = append(num2str(i),c,theFiles2(k).name);

        pic = imread(fullfile(folder_pics_2, theFiles2(k).name));
        [x,y,z] = size(pic);

        new_pic = imresize(pic, [rows, cols]);

        new_pic = cast(new_pic,'uint8');
        imwrite(new_pic, append(X_foldert_save_to, file_name))

        i = i + 1;
    end
end
fprintf('2020 done \n')

%% saving files
table_files = table(name(1:i-1), target(1:i-1));
writetable(table_files, y_save_to);
toc