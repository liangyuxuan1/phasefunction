clear all;
clc;

srcDataPath = 'imageCW_v3';   % path to the raw simulation results
dstDataPath = 'lineCW_v3';     % path to store the profiles
if ~exist(dstDataPath, 'dir')
    mkdir(dstDataPath);
end

changeDataList(srcDataPath, 'trainDataCW_v3_image.csv', dstDataPath);
changeDataList(srcDataPath, 'testDataCW_v3_image.csv',  dstDataPath);
% copyfile([srcDataPath, filesep, '*.csv'], dstDataPath);

function changeDataList(srcDataPath, dataListFileName, dstDataPath)
    T = readtable(fullfile(srcDataPath, dataListFileName));
    % varNames = {'image', 'ua', 'us', 'g', 'photonPosNum'};
    varTypes = {'string', 'double', 'double', 'double', 'int32'};
    newT = table('Size', [height(T)*2, 5],  'VariableTypes',varTypes, 'VariableNames', T.Properties.VariableNames);
    
    for i = 1:height(T)
        filename = cell2mat(table2array(T(i, 1)));
        sepPos = strfind(filename, filesep);
        subDir = filename(1:sepPos-1);

        imgFileName = fullfile(srcDataPath, filename)
        img = load(imgFileName).rawData;
        if isempty(img) % file not exist
            continue;
        end
        
        dstImagePath = fullfile(dstDataPath, subDir);
        if ~exist(dstImagePath, 'dir')
            mkdir(dstImagePath);
        end
            
        [M, N] = size(img);
        line = img(M/2+1, :) + img(M/2, :);
        dstFileName = strcat(filename(1:end-4), '_1.mat');
        newT(2*i-1, :) = T(i, :);
        newT(2*i-1, 1) = {dstFileName};
        newT(2*i-1, 5) = {sum(line>0)};
        save(fullfile(dstDataPath, dstFileName), 'line');
        
        line = (img(:, N/2+1) + img(:, N/2))';
        dstFileName = strcat(filename(1:end-4), '_2.mat');
        newT(2*i, :) = T(i, :);
        newT(2*i, 1) = {dstFileName};
        newT(2*i, 5) = {sum(line>0)};        
        save(fullfile(dstDataPath, dstFileName), 'line');
    end % for table
end

