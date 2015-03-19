load('/Users/nishithkhandwala/Desktop/CS199/syllable_dataset.mat');

for BIT_INDEX = 1:172
    if BIT_INDEX < 10
        fileName = strcat('BIT00',num2str(BIT_INDEX));
    elseif BIT_INDEX >= 10 && BIT_INDEX < 100
        fileName = strcat('BIT0',num2str(BIT_INDEX));
    elseif BIT_INDEX >= 100
        fileName = strcat('BIT',num2str(BIT_INDEX));
    end
       
    fileNameEval = eval(fileName);
    structData = fileNameEval.F;
    
    fileName = strcat(fileName, '.mat');
    save(fileName,'structData')
end

for HA_INDEX = 1:103
    if HA_INDEX < 10
        fileName = strcat('HA00',num2str(HA_INDEX));
    elseif HA_INDEX >= 10 && HA_INDEX < 100
        fileName = strcat('HA0',num2str(HA_INDEX));
    elseif HA_INDEX >= 100
        fileName = strcat('HA',num2str(HA_INDEX));
    end
       
    fileNameEval = eval(fileName);
    structData = fileNameEval.F;
    
    fileName = strcat(fileName, '.mat');
    save(fileName,'structData')
end

for HABIT_INDEX = 1:138
    if HABIT_INDEX < 10
        fileName = strcat('HABIT00',num2str(HABIT_INDEX));
    elseif HABIT_INDEX >= 10 && HABIT_INDEX < 100
        fileName = strcat('HABIT0',num2str(HABIT_INDEX));
    elseif HABIT_INDEX >= 100
        fileName = strcat('HABIT',num2str(HABIT_INDEX));
    end
       
    fileNameEval = eval(fileName);
    structData = fileNameEval.F;
    
    fileName = strcat(fileName, '.mat');
    save(fileName,'structData')
end

for NAL_INDEX = 1:170
    if NAL_INDEX < 10
        fileName = strcat('NAL00',num2str(NAL_INDEX));
    elseif NAL_INDEX >= 10 && NAL_INDEX < 100
        fileName = strcat('NAL0',num2str(NAL_INDEX));
    elseif NAL_INDEX >= 100
        fileName = strcat('NAL',num2str(NAL_INDEX));
    end
       
    fileNameEval = eval(fileName);
    structData = fileNameEval.F;
    
    fileName = strcat(fileName, '.mat');
    save(fileName,'structData')
end

for SIG_INDEX = 1:102
    if SIG_INDEX < 10
        fileName = strcat('SIG00',num2str(SIG_INDEX));
    elseif SIG_INDEX >= 10 && SIG_INDEX < 100
        fileName = strcat('SIG0',num2str(SIG_INDEX));
    elseif SIG_INDEX >= 100
        fileName = strcat('SIG',num2str(SIG_INDEX));
    end
       
    fileNameEval = eval(fileName);
    structData = fileNameEval.F;
    
    fileName = strcat(fileName, '.mat');
    save(fileName,'structData')
end

for SIGNAL_INDEX = 1:148
    if SIGNAL_INDEX < 10
        fileName = strcat('SIGNAL00',num2str(SIGNAL_INDEX));
    elseif SIGNAL_INDEX >= 10 && SIGNAL_INDEX < 100
        fileName = strcat('SIGNAL0',num2str(SIGNAL_INDEX));
    elseif SIGNAL_INDEX >= 100
        fileName = strcat('SIGNAL',num2str(SIGNAL_INDEX));
    end
       
    fileNameEval = eval(fileName);
    structData = fileNameEval.F;
    
    fileName = strcat(fileName, '.mat');
    save(fileName,'structData')
end
