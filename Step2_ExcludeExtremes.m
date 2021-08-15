clear all;
clc;

srcDataPath = 'imageCW_v3';     % path to the datalist
dstDataPath = 'imageCW_v3';     % path to store the reduced datalist

changeDataList(srcDataPath, 'trainDataCW_v3_image.csv', 'trainDataCW_v3_ExcludeExtremes.csv', 80);
changeDataList(srcDataPath, 'testDataCW_v3_image.csv',  'valDataCW_v3_ExcludeExtremes.csv', 100);

changeDataList(srcDataPath, 'trainDataCW_v3_image.csv', 'trainDataCW_v3_ExcludeExtremes_small.csv', 40);
changeDataList(srcDataPath, 'testDataCW_v3_image.csv',  'valDataCW_v3_ExcludeExtremes_small.csv', 90);

function changeDataList(srcDataPath, srcDataListFileName, dstDataListFileName, imgNumThreshold)
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

    filenames = cell2mat(table2array(T(:, 1)));
    dotPos = strfind(filenames(1,:), '.');
    imgNums = filenames(:, dotPos(1)-4:dotPos(1)-1);
    imgNums = str2num(imgNums);
    idx = find(imgNums > imgNumThreshold);
    
    T(idx,:) = [];  % keep only part of images
    
    writetable(T, fullfile(srcDataPath, dstDataListFileName));
end

