% Load your .mat file
load('flavio_eeg_data/10501_011119.mat');

% Print Rat ID and Session Date (should be char or numeric arrays)
disp('Rat ID:');
disp(Data_eeg{1});  % Usually a string/number

disp('Session Date:');
disp(Data_eeg{2});

% Print EEG matrix shape
fprintf('EEG shape: (%d, %d)\n', size(Data_eeg{3}, 1), size(Data_eeg{3}, 2));
fprintf('EEG time shape: (%d, %d)\n', size(Data_eeg{4}, 1), size(Data_eeg{4}, 2));
fprintf('Velocity trace shape: (%d, %d)\n', size(Data_eeg{5}, 1), size(Data_eeg{5}, 2));
fprintf('Velocity time shape: (%d, %d)\n', size(Data_eeg{6}, 1), size(Data_eeg{6}, 2));
fprintf('NM peak times shape: (%d, %d)\n', size(Data_eeg{7}, 1), size(Data_eeg{7}, 2));
fprintf('NM sizes shape: (%d, %d)\n', size(Data_eeg{8}, 1), size(Data_eeg{8}, 2));
fprintf('ITI peak times shape: (%d, %d)\n', size(Data_eeg{9}, 1), size(Data_eeg{9}, 2));
fprintf('ITI sizes shape: (%d, %d)\n', size(Data_eeg{10}, 1), size(Data_eeg{10}, 2));

% Show first 5 EEG values (row 1)
disp('EEG (ch 1, first 5):');
disp(Data_eeg{3}(1, 1:5));

disp('EEG time (first 5):');
disp(Data_eeg{4}(1, 1:5));


disp('Velocity trace (first 5):');
disp(Data_eeg{5}(1:5)); 

disp('NM sizes (first 10):');
disp(Data_eeg{8}(1:10)); 

disp('ITI sizes (first 10):');
disp(Data_eeg{10}(1:10)); 
