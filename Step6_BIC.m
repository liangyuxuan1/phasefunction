% calc the BIC of Resnet18 with different number of Gaussian components

% https://pypi.org/project/RegscorePy/

% n = the number of data points
% k = the number of free parameters to be estimated
% L = the maximized value of the likelihood function for the estimated model
% Under the assumption that the model errors or disturbances are normally distributed
% BIC = n*ln(RSS/n) + k*ln(n)
% RSS: residual sum of squares. RSS/n = MSE

nComponent = 3:10;
MSE = [0.011714, 0.005963, 0.007559, 0.006093, 0.005346, 0.005211, 0.004737, 0.005002];
n = 6000; % trainDataCW_v4
nChLastLayerResNet = 512;
k = nChLastLayerResNet*nComponent*3 +3*nComponent;
BIC = n*log(MSE) + k*log(n)

AIC = n*log(MSE) + 2*k