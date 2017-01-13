disp('clearing...');
clear;
disp('done');

disp('loading data...');
dataset_file = 'dataset_1.txt';
fid = fopen(dataset_file);

num_str = fgetl(fid);
image_count = str2num(num_str);

dataset_blurred = {};
dataset_sharp = {};

for i = 1:image_count
    photo_file = fgetl(fid);
    label_file = fgetl(fid);
    
    photo = imread(photo_file);
    labels = imread(label_file);
    
    [new_blurred, new_sharp] = dataset_from_photo(photo, labels, 5);
    
    dataset_blurred = {dataset_blurred{:}, new_blurred{:}};
    dataset_sharp   = {dataset_sharp{:},   new_sharp{:}};
end;

fclose(fid);
disp('done');

disp('merging...');
count_blurred = size(dataset_blurred, 2);
count_sharp = size(dataset_sharp, 2);
drop_step = count_blurred / count_sharp;

new_counter = 0;
for i = 1:drop_step:count_blurred
	new_counter = new_counter + 1;
    dataset_blurred_small{new_counter} = dataset_blurred{floor(i)};
end;
count_blurred_small = new_counter;

clearvars dataset_blurred;
disp('done');

disp('shuffling...');
shuffle_steps = count_sharp * 2;
for i = 1:shuffle_steps
    rand_i = floor(rand() * count_sharp) + 1;
    rand_j = floor(rand() * count_sharp) + 1;
    
    tmp = dataset_sharp{rand_i};
    dataset_sharp{rand_i} = dataset_sharp{rand_j};
    dataset_sharp{rand_j} = tmp;
end;

shuffle_steps = count_blurred_small * 2;
for i = 1:shuffle_steps
    rand_i = floor(rand() * count_blurred_small) + 1;
    rand_j = floor(rand() * count_blurred_small) + 1;
    
    tmp = dataset_blurred_small{rand_i};
    dataset_blurred_small{rand_i} = dataset_blurred_small{rand_j};
    dataset_blurred_small{rand_j} = tmp;
end;
disp('done');

disp('saving...');
batch_size = 50;
batch_number = floor(count_sharp / (batch_size / 2));

batch_folder = ['batches_', datestr(clock)];
batch_folder(batch_folder(:) == ':') = '-';
batch_folder(batch_folder(:) == ' ') = '_';
mkdir(batch_folder);

fsummary = fopen([batch_folder, '/info.txt'], 'w');
fprintf(fsummary, '%d\n', batch_number);

for i = 1:batch_number
    filename = [batch_folder, '/', 'batch_', num2str(i), '.data'];
    fprintf(fsummary, [filename, '\n']);
    
    fid = fopen(filename, 'w');
    
    disp(filename);
    
    for j = 1:batch_size / 2
        fwrite(fid, 1, 'uint8');
        fwrite(fid, 0, 'uint8');
        fwrite(fid, dataset_sharp{(i - 1) * (batch_size / 2) + j}.photo(:), 'uint8');
    end;
    
    for j = 1:batch_size / 2
        fwrite(fid, 0, 'uint8');
        fwrite(fid, 1, 'uint8');
        fwrite(fid, dataset_blurred_small{(i - 1) * (batch_size / 2) + j}.photo(:), 'uint8');
    end;
    
    fclose(fid);
end;

fclose(fsummary);
disp('done');

disp('clearing...');
clear;
disp('done');





