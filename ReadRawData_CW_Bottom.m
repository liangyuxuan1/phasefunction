function rawData = ReadRawData(rawDataFileName)
% read raw data into matrix and normalize it.

fid = fopen(rawDataFileName);
if fid < 0
    rawData = [];
    return;
end

flag = 0;
while 1
    aline = fgetl(fid);
    if strncmp(aline, 'CountX CountY', 13)
        flag = flag + 1;
    end
    if flag == 2
        context = textscan(aline,'%s %s %d %d');
        imgSizeX = context{3};
        imgSizeY = context{4};
        
        aline = fgetl(fid);
        rawData = fscanf(fid,'%f', imgSizeX*imgSizeY);
        rawData = reshape(rawData, [imgSizeY, imgSizeX]);

        break;
    end
end % while
fclose(fid);