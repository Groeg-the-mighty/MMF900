
save_to = "Z:\Documents\MMF900\2020_with_2019malignant_PY.csv";

matrix = readmatrix("2020_with_2019malignant.csv");

zero = zeros(1,45000);
matrix2 = [zero;matrix];

writematrix(matrix2, save_to);