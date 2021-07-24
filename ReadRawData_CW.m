function rawData = ReadRawData(rawDataFileName)
% read raw data into matrix and normalize it.

fid = fopen(rawDataFileName);
if fid < 0
    rawData = [];
    return;
end

while 1
    flag = 0;
    
    aline = fgetl(fid);
    if strncmp(aline, 'CountX CountY', 13)
        context = textscan(aline,'%s %s %d %d');
        imgSizeX = context{3};
        imgSizeY = context{4};
        
        aline = fgetl(fid);
        rawData = fscanf(fid,'%f', imgSizeX*imgSizeY);
        rawData = reshape(rawData, [imgSizeY, imgSizeX]);
        flag = 1;
    end
    if flag == 1
        break;
    end
end % while
fclose(fid);