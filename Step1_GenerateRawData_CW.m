clear all;
clc;

phantomFile = 'newProject_CW_200.mse';

% parameter range, refer to: Brett H. Hokr1 and Joel N. Bixler2, Machine
% learning estimation of tissue optical properties, Scientific Reports,
% 11:6561, 2021
ua = 0.05;      % absorption coefficient, [0.01, 10] mm^-1
us = 10;        % scattering coefficient, [0.1, 100] mm^-1
n  = 1.3;       % refractive index, no need to vary for single layer slab
g  = [0.5:0.01:0.95];    % anistropic scattering coefficient of HG function
trainNum = 70;           % training number of runs (images) for each set of parameters
testNum  = 30;           % testing number of runs (images) for each set of parameters

dataPath = 'rawDataCW';   % path to store the raw simulation results
if ~exist(dataPath,'dir')
    mkdir(dataPath);
end

totalTrainNum = length(ua)*length(us)*length(g)*trainNum;
totalTestNum  = length(ua)*length(us)*length(g)*testNum;
varNames = {'image', 'ua', 'us', 'g', 'run'};
varTypes = {'string', 'double', 'double', 'double', 'double'};
trainTableCW = table('Size', [totalTrainNum,5], 'VariableTypes',varTypes,'VariableNames',varNames);
testTableCW  = table('Size', [totalTestNum,5],  'VariableTypes',varTypes,'VariableNames',varNames);

for ia = 1:length(ua)
    for is = 1:length(us)
        for ig = 1:length(g)
            samplePath = sprintf("g%03d", ig);
            if ~exist(fullfile(dataPath, samplePath), 'dir')
                mkdir(fullfile(dataPath, samplePath));
            end
            for i = 1:trainNum+testNum
                dataFileName = sprintf("a%03d_s%03d_g%03d_%04d",ia, is, ig, i);  % data file name
                
                if i > trainNum
                    counter = (ia-1)*length(us)*length(g)*testNum + (is-1)*length(g)*testNum + (ig-1)*testNum + i-trainNum;
                    testTableCW(counter,:)  = {strcat(fullfile(samplePath, dataFileName), '.T.CW.mat'), ua(ia), us(is), g(ig), i};
                else
                    counter = (ia-1)*length(us)*length(g)*trainNum + (is-1)*length(g)*trainNum + (ig-1)*trainNum + i;
                    trainTableCW(counter,:) = {strcat(fullfile(samplePath, dataFileName), '.T.CW.mat'), ua(ia), us(is), g(ig), i};
                end

                dataFileName = fullfile(dataPath, samplePath, dataFileName);
                parameters = ['MOSE\moseVCTest.exe', phantomFile, dataFileName, num2str(ua(ia)), num2str(us(is)), num2str(g(ig)), num2str(n)];
                cmdLine = strjoin(parameters, ' ');
                
                system(cmdLine);    % call exe

                % check the results
%                 dataFileName = strcat(dataFileName, '.T.CW');
%                 rawData = ReadRawData_CW(dataFileName);
%                 figure; imshow(rawData);
            end % of ig
        end % of in
    end % of is
end % of ia

writetable(trainTableCW, 'trainDataCW.csv');
writetable(testTableCW,  'testDataCW.csv');