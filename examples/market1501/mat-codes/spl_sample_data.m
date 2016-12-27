clc;clear all;close all;
load ('../datamat/trainData.mat');
n_Train = length(trainID);
unique_id = unique(trainID);
revers_id = zeros(max(unique_id), 1);
find_label = zeros(n_Train, 1);
for n = 1:length(unique_id)
    assert (unique_id(n) >= 1);
    revers_id( unique_id(n) ) = n;
end
count_num = zeros(length(unique_id), 1);
for n = 1:n_Train
	find_label(n) = revers_id(trainID(n));
	count_num(find_label(n)) = count_num(find_label(n)) + 1;
end

ratio = 0.2;
sample_per_class = cell(length(unique_id), 1);
for label = 1:length(unique_id)
	ids = find(find_label == label);
	spl = randperm(numel(ids), ceil(numel(ids)*ratio));
	spl = ids(spl);
	fprintf('Label %4d has %4d / %4d images\n', n, numel(spl), numel(ids));
	sample_per_class{label} = spl;
end
sampled = cat(1, sample_per_class{:});
sampled = sort(sampled);
fprintf('Total sampled %4d / %4d , ratio : %.3f\n', numel(sampled), n_Train, numel(sampled) / n_Train);

train_list_file = fopen('../lists/train.epoch.1', 'w');
for i = 1:numel(sampled)
	n = sampled(i);
    fprintf(train_list_file, '%s/%s %d\n', train_dir, train_files(n).name, find_label(n)-1);
end
fclose(train_list_file);

left = setdiff((1:n_Train), sampled);
train_list_file = fopen('../lists/train.epoch.1L', 'w');
for i = 1:numel(left)
	n = left(i);
    fprintf(train_list_file, '%s/%s %d\n', train_dir, train_files(n).name, find_label(n)-1);
end
fclose(train_list_file);
