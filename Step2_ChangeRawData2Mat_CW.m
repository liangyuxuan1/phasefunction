clear all;
clc;

srcDataPath = 'rawDataCW_FixedG_500';   % path to the raw simulation results
dstDataPath = 'imageCW_FixedG_500';     % path to store the images
if ~exist(dstDataPath, 'dir')
    mkdir(dstDataPath);
end

gDirList = dir(fullfile(srcDataPath, 'g*'));   % get the directory list
for i=1:length(gDirList)
    dstImagePath = fullfile(dstDataPath, gDirList(i).name)
    if ~exist(dstImagePath, 'dir')
        mkdir(dstImagePath);
    end
    
    rawDataList = dir(fullfile(srcDataPath, gDirList(i).name, '*.CW'));
    for j=1:length(rawDataList)
        rawDataFileName = fullfile(srcDataPath, gDirList(i).name, rawDataList(j).name);
        dstImageFileName = fullfile(dstImagePath, strcat(rawDataList(j).name, '.mat'));
        rawData = ReadRawData_CW(rawDataFileName);
        %imwrite(rawData, dstImageFileName);
        save(dstImageFileName, 'rawData');
    end % for j, images
end  % for i, dirlist

