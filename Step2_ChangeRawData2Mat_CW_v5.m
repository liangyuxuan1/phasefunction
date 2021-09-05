clear all;
clc;

srcDataPath = 'rawDataCW_v5';   % path to the raw simulation results
dstDataPath = 'imageCW_v5';     % path to store the images
if ~exist(dstDataPath, 'dir')
    mkdir(dstDataPath);
end

changeDataList(srcDataPath, 'DataListCW_v5.csv', dstDataPath);
copyfile([srcDataPath, filesep, '*.csv'], dstDataPath);

function changeDataList(srcDataPath, dataListFileName, dstDataPath)
    % readtable get correct csv in Matlab2019b
    % but will get NaN for string in Matlab2020a
    T = readtable(fullfile(srcDataPath, dataListFileName));
    
    for i = 1:height(T)
        filename = cell2mat(table2array(T(i, 1)));
        if isempty(filename)
            continue;
        end
        sepPos = strfind(filename, filesep);
        subDir = filename(1:sepPos-1);

        rawDataFileName = fullfile(srcDataPath, filename)
        rawData = ReadRawData_CW(rawDataFileName);
        if isempty(rawData) % file not exist
            continue;
        end
        
        dstImagePath = fullfile(dstDataPath, subDir);
        if ~exist(dstImagePath, 'dir')
            mkdir(dstImagePath);
        end
            
        T(i, 1) = {strcat(filename, '.mat')};
        dstImageFileName = fullfile(dstDataPath, strcat(filename, '.mat'));
        save(dstImageFileName, 'rawData');
    end % for table
end

