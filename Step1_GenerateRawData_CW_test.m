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
ua = [0.02, 0.08, 0.3, 0.8, 3, 6, 7.5, 9];      % absorption coefficient, [0.01, 10] mm^-1
us = [0.03, 0.08, 0.3, 0.8, 3, 6, 20, 60, 80];        % scattering coefficient, [0.1, 100] mm^-1
n  = 1.3;       % refractive index, no need to vary for single layer slab
g  = [-0.95:0.1:0.95];    % anistropic scattering coefficient of HG function
trainNum = 10;           % training number of runs (images) for each set of parameters
valNum  = 0;            % validation number of runs (images) for each set of parameters

dataPath = 'rawDataCW_test';   % path to store the raw simulation results
if ~exist(dataPath,'dir')
    mkdir(dataPath);
end

totalTrainNum = length(ua)*length(us)*length(g)*trainNum;
totalValNum  = length(ua)*length(us)*length(g)*valNum;
varNames = {'image', 'ua', 'us', 'g', 'photonPosNum'};
varTypes = {'string', 'double', 'double', 'double', 'int32'};
trainTableCW = table('Size', [totalTrainNum,5], 'VariableTypes',varTypes,'VariableNames',varNames);
valTableCW   = table('Size', [totalValNum,5],  'VariableTypes',varTypes,'VariableNames',varNames);
for ia = 1:length(ua)
    for is = 1:length(us)
        for ig = 1:length(g)
            samplePath = sprintf("a%02d_s%02d_g%03d",ia, is, ig);
            if ~exist(fullfile(dataPath, samplePath), 'dir')
                mkdir(fullfile(dataPath, samplePath));
            end
            for i = 1:trainNum+valNum
                dataFileName = sprintf("a%03d_s%03d_g%03d_%04d",ia, is, ig, i);  % data file name
                fullDataFileName = fullfile(dataPath, samplePath, dataFileName);
                parameters = ['MOSE\moseVCTest.exe', phantomFile, fullDataFileName, num2str(ua(ia)), num2str(us(is)), num2str(g(ig)), num2str(n)];
                cmdLine = strjoin(parameters, ' ');
                
                % check the results
                if ~isfile(strcat(fullDataFileName, '.T.CW'))
                    % system(cmdLine);    % call MOSE
                end
                
                photonNum = -1; % -1: not processed in step2
                
                dataFileName = strcat(dataFileName, '.T.CW');
                if i > trainNum
                    counter = (ia-1)*length(us)*length(g)*valNum + (is-1)*length(g)*valNum + (ig-1)*valNum + i-trainNum;
                    valTableCW(counter,:)  = {fullfile(samplePath, dataFileName), ua(ia), us(is), g(ig), photonNum};
                else
                    counter = (ia-1)*length(us)*length(g)*trainNum + (is-1)*length(g)*trainNum + (ig-1)*trainNum + i;
                    trainTableCW(counter,:) = {fullfile(samplePath, dataFileName), ua(ia), us(is), g(ig), photonNum};
                end
                % figure; imshow(binaryImg);

            end % of ig
        end % of in
    end % of is
end % of ia

writetable(trainTableCW, [dataPath, filesep, 'testDataCW_v4.csv']);
% writetable(valTableCW,   [dataPath, filesep, 'valDataCW_v4.csv']);