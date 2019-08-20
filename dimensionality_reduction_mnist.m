% Dimensionality reduction 
clear all; 

% Load dataset 
imageFileName = 't10k-images.idx3-ubyte';
labelFileName = 't10k-labels.idx1-ubyte';
% Please check your directory that has 'processMNISTdata.m'
% More friendly explanations can be found here https://www.mathworks.com/help/stats/visualize-high-dimensional-data-using-t-sne.html
addpath('C:\Users\user\Documents\MATLAB\Examples\stats\VisualizeHighDimensionalDataUsingTSNEExample'); % add to the path
[X,L] = processMNISTdata(imageFileName,labelFileName);
rmpath('C:\Users\user\Documents\MATLAB\Examples\stats\VisualizeHighDimensionalDataUsingTSNEExample'); % remove from the path
% X is a data matrix and L is a group label.  

% Number of data samples, dimension of data 
[n,p] = size(X); 

figure; 
% Linear method 
%%% Principal component analysis (PCA) 
% Centering 
X = X - repmat(mean(X,1),[n 1]); 
% Covariance matrix 
C = cov(X);
% Eigendecomposition of C 
[U,D] = eig(C); 
% Eigenvalues 
D = diag(D);
% Sort eigenvectors in the descending order of eigenvalues 
[tval,tind] = sort(D,'descend'); 
U = U(:,tind); 
D = tval;
% Projection to the principal components 
Y = X*U(:,1:2);

% Plot the results of PCA  
subplot(1,2,1); 
gscatter(Y(:,1),Y(:,2),L); 
title('PCA'); 
set(gca,'FontSize',14); 


% Nonlinear method 
%%% t-Distributed Stochastic Neighbor Embedding (tSNE) 
rng default % for reproducibility
Y = tsne(X,'Algorithm','barneshut','NumPCAComponents',50);

% Plot the results of tSNE
subplot(1,2,2); 
gscatter(Y(:,1),Y(:,2),L); 
title('tSNE'); 
set(gca,'FontSize',14); 

