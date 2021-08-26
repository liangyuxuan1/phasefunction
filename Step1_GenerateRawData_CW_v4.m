% 2021-08-23
% Simplify the problem by fixing the ua and us
% Use the ua an us of typical tissues and vary only g

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

% parameter at 650 nm
% T1 fat, heart, gut, liver, lung, kidney
pua = [0.00504, 0.07859, 0.01504, 0.47078, 0.26296, 0.08811];          % absorption coefficient, [0.01, 10] mm^-1
pus = [20.45447, 6.7104, 18.49734, 6.9999, 36.81805, 16.84647];         % scattering coefficient, [0.1, 100] mm^-1
% g = [0.94, 0.85, 0.92, 0.9, 0.94, 0.86]

n  = 1.37;               % refractive index, no need to vary for single layer slab
g  = [0.55:0.1:0.95];    % anistropic scattering coefficient of HG function
trainNum = 200;           % training number of runs (images) for each set of parameters
valNum  = 30;            % validation number of runs (images) for each set of parameters

dataPath = 'rawDataCW_v4';   % path to store the raw simulation results
if ~exist(dataPath,'dir')
    mkdir(dataPath);
end

totalTrainNum = length(pua)*length(g)*trainNum;
totalValNum  = length(pua)*length(g)*valNum;
varNames = {'image', 'ua', 'us', 'g', 'photonPosNum'};
varTypes = {'string', 'double', 'double', 'double', 'int32'};
trainTableCW = table('Size', [totalTrainNum,5], 'VariableTypes',varTypes,'VariableNames',varNames);
valTableCW   = table('Size', [totalValNum,5],  'VariableTypes',varTypes,'VariableNames',varNames);
for p = 1:6
    ua = pua(p);
    us = pus(p);
        for ig = 1:length(g)
            samplePath = sprintf("p%02d_g%03d",p, ig);
            if ~exist(fullfile(dataPath, samplePath), 'dir')
                mkdir(fullfile(dataPath, samplePath));
            end
            for i = 1:trainNum+valNum
                dataFileName = sprintf("p%02d_g%03d_%04d", p, ig, i);  % data file name
                fullDataFileName = fullfile(dataPath, samplePath, dataFileName);
                parameters = ['MOSE\moseVCTest.exe', phantomFile, fullDataFileName, num2str(ua), num2str(us), num2str(g(ig)), num2str(n)];
                cmdLine = strjoin(parameters, ' ');
                
                % check the results
                if ~isfile(strcat(fullDataFileName, '.T.CW'))
                    system(cmdLine);    % call MOSE
                end
                
                photonNum = -1; % -1: not processed in step2
                
                dataFileName = strcat(dataFileName, '.T.CW');
                if i > trainNum
                    counter = (p-1)*length(g)*valNum + (ig-1)*valNum + i-trainNum;
                    valTableCW(counter,:)  = {fullfile(samplePath, dataFileName), ua, us, g(ig), photonNum};
                else
                    counter = (p-1)*length(g)*trainNum + (ig-1)*trainNum + i;
                    trainTableCW(counter,:) = {fullfile(samplePath, dataFileName), ua, us, g(ig), photonNum};
                end
                % figure; imshow(binaryImg);
            end % of in
        end % of ig
end % of ia

writetable(trainTableCW, [dataPath, filesep, 'trainDataCW_v4.csv']);
writetable(valTableCW,   [dataPath, filesep, 'valDataCW_v4.csv']);