clear all;
clc;

srcDataPath = 'imageCW_v3';     % path to the datalist
dstDataPath = 'imageCW_v3';     % path to store the reduced datalist

changeDataList(srcDataPath, 'trainDataCW_v3.csv', 'trainDataCW_v3_ExcludeExtremes_PositiveG.csv');
changeDataList(srcDataPath, 'valDataCW_v3.csv',   'valDataCW_v3_ExcludeExtremes_PositiveG.csv');

srcDataPath = 'imageCW_v3_test';     % path to the datalist
dstDataPath = 'imageCW_v3_test';     % path to store the reduced datalist
changeDataList(srcDataPath, 'testDataCW_v3.csv',   'testDataCW_v3_ExcludeExtremes_PositiveG.csv');

function changeDataList(srcDataPath, srcDataListFileName, dstDataListFileName)
    T = readtable(fullfile(srcDataPath, srcDataListFileName));
    
    num_pixels = table2array(T(:, 5));
    idx = find(num_pixels < 100);
    
    filenames = cell2mat(table2array(T(idx, 1)));
    sepPos = strfind(filenames(1,:), filesep);
    subDirs = filenames(:, 1:sepPos-1);
    [dirNames, ia, ic] = unique(subDirs, 'rows');
    for i=1:length(dirNames)
        dirNum = length(find(ic == i));
        fprintf('%s: %d\n', dirNames(i,:), dirNum);
    end

    T(idx,:) = [];  % remove extreme images

    g = table2array(T(:, 4));
    idx = find(g < 0);
    T(idx,:) = [];  % keep only images with g>=0
    
    writetable(T, fullfile(srcDataPath, dstDataListFileName));
end

