clear all;
clc;

phantomFile = 'newProject_CW_500.mse';

% parameter range, refer to: Brett H. Hokr1 and Joel N. Bixler2, Machine
% learning estimation of tissue optical properties, Scientific Reports,
% 11:6561, 2021
% ua = 0.05;      % absorption coefficient, [0.01, 10] mm^-1
% us = 10;        % scattering coefficient, [0.1, 100] mm^-1
T = readtable('materials.csv');
us=table2array(T(:,2));
ua=table2array(T(:,3));
g=table2array(T(:,4));
n  = 1.3;       % refractive index, no need to vary for single layer slab
% g  = [-1:0.025:1];    % anistropic scattering coefficient of HG function
trainNum = 100;           % training number of runs (images) for each set of parameters
testNum  = 30;           % testing number of runs (images) for each set of parameters

dataPath = 'rawDataCW';   % path to store the raw simulation results
if ~exist(dataPath,'dir')
    mkdir(dataPath);
end

totalTrainNum = length(g)*trainNum;
totalTestNum  = length(g)*testNum;
varNames = {'image', 'ua', 'us', 'g'};
varTypes = {'string', 'double', 'double', 'double'};
trainTableCW = table('Size', [totalTrainNum,4], 'VariableTypes',varTypes,'VariableNames',varNames);
testTableCW  = table('Size', [totalTestNum,4],  'VariableTypes',varTypes,'VariableNames',varNames);

for ig = 1:length(g)
    samplePath = sprintf("g%03d", ig);
    if ~exist(fullfile(dataPath, samplePath), 'dir')
        mkdir(fullfile(dataPath, samplePath));
    end
    for i = 1:trainNum+testNum
        dataFileName = sprintf("g%02d_%04d",ig,i);  % data file name
        
        if i > trainNum
            counter = (ig-1)*testNum + i - trainNum;
            testTableCW(counter,:)  = {strcat(fullfile(samplePath, dataFileName), '.T.CW.mat'), ua(ig), us(ig), g(ig)};
        else
            counter = (ig-1)*trainNum + i;
            trainTableCW(counter,:) = {strcat(fullfile(samplePath, dataFileName), '.T.CW.mat'), ua(ig), us(ig), g(ig)};
        end
        
        dataFileName = fullfile(dataPath, samplePath, dataFileName);
        parameters = ['MOSE\moseVCTest.exe', phantomFile, dataFileName, num2str(ua(ig)), num2str(us(ig)), num2str(g(ig)), num2str(n)];
        cmdLine = strjoin(parameters, ' ');
        
%         system(cmdLine);    % call exe
        
        % check the results
        %                 dataFileName = strcat(dataFileName, '.T.CW');
        %                 rawData = ReadRawData_CW(dataFileName);
        %                 figure; imshow(rawData);
    end % of ig
end % of in

writetable(trainTableCW, [dataPath, filesep, 'trainDataCW.csv']);
writetable(testTableCW,  [dataPath, filesep, 'testDataCW.csv']);