% Train Convolutional Neural Network for Regression

clear all;
close all;
clc;

% The data set contains synthetic images of handwritten digits 
% together with the corresponding angles (in degrees) by which each image is rotated
% [XTrain,~,YTrain] = digitTrain4DArrayData;
% [XValidation,~,YValidation] = digitTest4DArrayData;

% read the data list
trainDataCSV = 'imageCW\trainDataCW.csv';
trainDataTable = readtable( trainDataCSV, 'Delimiter', ',' );
trainImageList = fullfile(pwd, 'imageCW', trainDataTable.image);
trainImds = imageDatastore(trainImageList, 'LabelSource', 'foldernames', 'FileExtensions', '.mat', 'ReadFcn', @load );
[XTrain, ~] = imds2array(trainImds);
YTrain = trainDataTable.g;

testDataCSV = 'imageCW\testDataCW.csv';
testDataTable = readtable( testDataCSV, 'Delimiter', ',' );
testImageList = fullfile(pwd, 'imageCW', testDataTable.image);
testImds = imageDatastore(testImageList, 'LabelSource', 'foldernames', 'FileExtensions', '.mat', 'ReadFcn', @load );
[XValidation, ~] = imds2array(testImds);
YValidation = testDataTable.g;

% Display 20 random training images using imshow.
numTrainImages = numel(YTrain);
figure
idx = randperm(numTrainImages,20);
for i = 1:numel(idx)
    subplot(4,5,i)    
    imshow(log(XTrain(:,:,:,idx(i))+1), [])
    drawnow
end

imageHeight = size(XTrain,1);
imageWidth  = size(XTrain,2);
imageChannel= size(XTrain,3);
% To solve the regression problem, create the layers of the network and include a regression layer at the end of the network.
layers = [
    imageInputLayer([imageHeight imageWidth imageChannel])

    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    averagePooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    averagePooling2dLayer(2,'Stride',2)
  
    convolution2dLayer(3,64,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    averagePooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,64,'Padding','same')
    batchNormalizationLayer
    reluLayer

    averagePooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,64,'Padding','same')
    batchNormalizationLayer
    reluLayer

    averagePooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,128,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    globalAveragePooling2dLayer
    
    fullyConnectedLayer(1)
    regressionLayer];

% Train the network
miniBatchSize  = 32;
validationFrequency = floor(numel(YTrain)/miniBatchSize);
options = trainingOptions('sgdm', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',10, ...
    'InitialLearnRate',1e-3, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.1, ...
    'LearnRateDropPeriod',20, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{XValidation,YValidation}, ...
    'ValidationFrequency',validationFrequency, ...
    'Plots','training-progress', ...
    'Verbose',true);

net = trainNetwork(XTrain,YTrain,layers,options);
net.Layers

% Test the network
YPredicted = predict(net,XValidation, 'ExecutionEnvironment','cpu');
predictionError = YValidation - YPredicted;

thr = 0.001;
numCorrect = sum(abs(predictionError) < thr);
numValidationImages = numel(YValidation);

accuracy = numCorrect/numValidationImages

squares = predictionError.^2;
rmse = sqrt(mean(squares))