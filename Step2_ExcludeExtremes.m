clear all;
clc;

srcDataPath = 'imageCW_v3';     % path to the datalist
dstDataPath = 'imageCW_v3';     % path to store the reduced datalist

changeDataList(srcDataPath, 'trainDataCW_v3_image.csv', 'trainDataCW_v3_ExcludeExtremes.csv');
changeDataList(srcDataPath, 'testDataCW_v3_image.csv',  'valDataCW_v3_ExcludeExtremes.csv');

function changeDataList(srcDataPath, srcDataListFileName, dstDataListFileName)
    T = readtable(fullfile(srcDataPath, srcDataListFileName));
    
    num_pixels = table2array(T(:, 5));
    idx = find(num_pixels < 100);
    
    filename = cell2mat(table2array(T(idx, 1)));
    sepPos = strfind(filename(1,:), filesep);
    subDirs = filename(:, 1:sepPos-1);
    [dirNames, ia, ic] = unique(subDirs, 'rows');
    for i=1:length(dirNames)
        dirNum = length(find(ic == i));
        fprintf('%s: %d\n', dirNames(i,:), dirNum);
    end

    T(idx,:) = [];
    writetable(T, fullfile(srcDataPath, dstDataListFileName));
end

