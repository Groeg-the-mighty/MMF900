function C = COVARIANCE(X)
% - Computation of covariance for large matrix
% This program is suited to compute covariance matrices for large matrices 
% that would cause the built in matlab function to return "out of memory" error. 
% The method is to break down the matrix into blocks that Matlab can handle 
% and then manually compute the covariance matrix block by block. 


% Change to higher value if out of memory
N_blocks = 10; % specify the number of blocks


%% function
[mx,nx] = size(X);
if mod(nx,N_blocks) ~= 0 % nx must be multiple of N_blocks
    error(message('# of blocks does not cover entire matrix. Change N_blocks')); 
end 

xc = X - sum(X,1)./mx;  % Remove mean
b_l = nx/N_blocks;

for i = 1:N_blocks
    for j = i:N_blocks
        C( (i-1)*b_l+1:i*b_l , (j-1)*b_l+1:j*b_l ) = xc(:, (i-1)*b_l+1:i*b_l)' * xc(:, (j-1)*b_l+1:j*b_l);
        if i ~= j
            C( (j-1)*b_l+1:j*b_l , (i-1)*b_l+1:i*b_l ) = C( (i-1)*b_l+1:i*b_l , (j-1)*b_l+1:j*b_l );
        end
    end
end

