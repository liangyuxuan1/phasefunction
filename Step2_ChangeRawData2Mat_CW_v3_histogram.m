clear all;
clc;

srcDataPath = 'imageCW_v3';   % path to the raw simulation results
dstDataPath = 'imageCW_v3_hist';     % path to store the histogram
if ~exist(dstDataPath, 'dir')
    mkdir(dstDataPath);
end

changeDataList(srcDataPath, 'trainDataCW_v3_ExcludeExtremes.csv', dstDataPath);
changeDataList(srcDataPath, 'valDataCW_v3_ExcludeExtremes.csv',  dstDataPath);
copyfile([srcDataPath, filesep, '*.csv'], dstDataPath);

function changeDataList(srcDataPath, dataListFileName, dstDataPath)
    T = readtable(fullfile(srcDataPath, dataListFileName));
    
    for i = 1:height(T)
        bProcessed = table2array(T(i, 5));
        
        filename = cell2mat(table2array(T(i, 1)));
        sepPos = strfind(filename, filesep);
        subDir = filename(1:sepPos-1);

        rawDataFileName = fullfile(srcDataPath, filename)
        load(rawDataFileName);
        [M, N] = size(rawData);
        binaryImg = rawData > 0;
        [X, Y] = find(binaryImg);
        R = sqrt((X-(M+1)/2).^2+(Y-(N+1)/2).^2);
        maxR = sqrt(M*M/4 + N*N/4);
        hR = maxR/100;
        binR = hR/2:hR:maxR;
        histR = hist(R, binR);
        
        dstImagePath = fullfile(dstDataPath, subDir);
        if ~exist(dstImagePath, 'dir')
            mkdir(dstImagePath);
        end
        
        dstImageFileName = fullfile(dstDataPath, strcat(filename, '.mat'));
        save(dstImageFileName, 'rawData');
    end % for table
end

