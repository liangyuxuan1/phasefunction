clear all;
close all;
clc;

phantomFile = 'newProject_CW_500.mse';

% parameters from inverse scattering paper, only 14 materials
% T = readtable('materials.csv');
% us=table2array(T(:,2));
% ua=table2array(T(:,3));
% g=table2array(T(:,4));

% parameter range, refer to: Brett H. Hokr1 and Joel N. Bixler2, Machine
% learning estimation of tissue optical properties, Scientific Reports,
% 11:6561, 2021
ua = [0.001, 0.05, 0.1, 0.5, 1, 2, 5, 10];      % absorption coefficient, [0.01, 10] mm^-1
us = [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100];        % scattering coefficient, [0.1, 100] mm^-1
n  = 1.3;       % refractive index, no need to vary for single layer slab
g  = [-1:0.1:1];    % anistropic scattering coefficient of HG function
trainNum = 80;           % training number of runs (images) for each set of parameters
testNum  = 20;            % testing number of runs (images) for each set of parameters

dataPath = 'rawDataCW';   % path to store the raw simulation results
if ~exist(dataPath,'dir')
    mkdir(dataPath);
end

totalTrainNum = length(ua)*length(us)*length(g)*trainNum;
totalTestNum  = length(ua)*length(us)*length(g)*testNum;
varNames = {'image', 'ua', 'us', 'g', 'ok'};
varTypes = {'string', 'double', 'double', 'double', 'logical'};
trainTableCW = table('Size', [totalTrainNum,5], 'VariableTypes',varTypes,'VariableNames',varNames);
testTableCW  = table('Size', [totalTestNum,5],  'VariableTypes',varTypes,'VariableNames',varNames);
for ia = 8 % 1:length(ua)
    for is = 1:length(us)
        for ig = 1:length(g)
            samplePath = sprintf("a%02d_s%02d_g%03d",ia, is, ig);
            if ~exist(fullfile(dataPath, samplePath), 'dir')
                mkdir(fullfile(dataPath, samplePath));
            end
            for i = 1:trainNum+testNum
                dataFileName = sprintf("a%03d_s%03d_g%03d_%04d",ia, is, ig, i);  % data file name
                dataFileName = fullfile(dataPath, samplePath, dataFileName);
                parameters = ['MOSE\moseVCTest.exe', phantomFile, dataFileName, num2str(ua(ia)), num2str(us(is)), num2str(g(ig)), num2str(n)];
                cmdLine = strjoin(parameters, ' ');
                
                % check the results
                dataFileName = strcat(dataFileName, '.T.CW')
                if ~isfile(dataFileName)
                    system(cmdLine);    % call MOSE
                end

                rawData = ReadRawData_CW(dataFileName);
               
                binaryImg = rawData > 0;
                if (sum(sum(binaryImg)) > 50*50)    % only use images with enought information
                    if i > trainNum
                        counter = (ia-1)*length(us)*length(g)*testNum + (is-1)*length(g)*testNum + (ig-1)*testNum + i-trainNum;
                        testTableCW(counter,:)  = {fullfile(samplePath, dataFileName), ua(ia), us(is), g(ig), 1};
                    else
                        counter = (ia-1)*length(us)*length(g)*trainNum + (is-1)*length(g)*trainNum + (ig-1)*trainNum + i;
                        trainTableCW(counter,:) = {fullfile(samplePath, dataFileName), ua(ia), us(is), g(ig), 1};
                    end
                    % figure; imshow(binaryImg);
                else
                    if i > trainNum
                        counter = (ia-1)*length(us)*length(g)*testNum + (is-1)*length(g)*testNum + (ig-1)*testNum + i-trainNum;
                        testTableCW(counter,:)  = {fullfile(samplePath, dataFileName), ua(ia), us(is), g(ig), 0};
                    else
                        counter = (ia-1)*length(us)*length(g)*trainNum + (is-1)*length(g)*trainNum + (ig-1)*trainNum + i;
                        trainTableCW(counter,:) = {fullfile(samplePath, dataFileName), ua(ia), us(is), g(ig), 0};
                    end
                end
            end % of ig
        end % of in
    end % of is
end % of ia

writetable(trainTableCW, [dataPath, filesep, 'trainDataCW.csv']);
writetable(testTableCW,  [dataPath, filesep, 'testDataCW.csv']);