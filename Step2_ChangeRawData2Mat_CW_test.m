clear all;
clc;

srcDataPath = 'rawDataCW_test';   % path to the raw simulation results
dstDataPath = 'imageCW_test';     % path to store the images
if ~exist(dstDataPath, 'dir')
    mkdir(dstDataPath);
end

changeDataList(srcDataPath, 'testDataCW_v4.csv',  dstDataPath);
copyfile([srcDataPath, filesep, '*.csv'], dstDataPath);

function changeDataList(srcDataPath, dataListFileName, dstDataPath)
    T = readtable(fullfile(srcDataPath, dataListFileName));
    
    bSave = 0;  % need save the datalist?
    for i = 1:height(T)
        bProcessed = table2array(T(i, 5));
        if ~isequal(bProcessed, -1) % -1: not been processed
            continue;
        end
        
        filename = cell2mat(table2array(T(i, 1)));
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
            
        bSave = 1;
        [M, N] = size(rawData);
        binaryImg = rawData > 0;
        T(i, 5) = {sum(sum(binaryImg))};
        T(i, 1) = {strcat(filename, '.mat')};
        dstImageFileName = fullfile(dstDataPath, strcat(filename, '.mat'));
        save(dstImageFileName, 'rawData');
        
        if mod(i, 400)==0
            writetable(T, fullfile(srcDataPath, dataListFileName));
        end
    end % for table
    
    if bSave
        writetable(T, fullfile(srcDataPath, dataListFileName));
    end
end

