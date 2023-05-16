clear all
clc

folder_pics = "C:\Users\lulle\Documents\MMF900\ISIC_2020_Training_JPEG\ISIC_2020_Training_Input_Resized\";        % Path : Folder with all training pictures
folder_info = "C:\Users\lulle\Documents\MMF900\ISIC_2020_Training_GroundTruth.csv";   % Path : Metadata for training pictures
save_to = "C:\Users\lulle\Documents\MMF900\training_short.csv";                       % Path : Name of document for new shorter metadata                                                                           

tic
filePattern = fullfile(folder_pics, '*.jpg');   % Returns directory with specific ending with *.jpg
theFiles = dir(filePattern);                    % Extracts all files as a struct 
size_arr = length(theFiles); 
target = zeros(size_arr,1);                     % Empty array for target values of remaining photos
name = string(target);                          % String array in correct dimensions
info = readtable(folder_info);                  % Read table from metadata 
i=1;

for k = 1 : length(info.target)
    f_name = append("resize_", info.image_name{k},'.jpg');         % Extend csv filename with .jpg for comparison   
    cell = struct2cell(theFiles);                                  % Create cell 
    exist = find(string(cell(1,:)) == string(f_name));             % Finds the photo in the metadata
        if( ~isempty(exist) )
            name(i) = f_name;
            target(i) = info.target(k);
            i = i + 1;
        end
end
table = table(name, target);
writetable(table, save_to);
toc