clc;
clear;
f = fopen('/media/Drive2/sagar/EGTEA_Gaze_Plus/action_annotation/train_split1.txt', 'r');
videos = textscan(f, '%s %d32 %d32', 'Delimiter', '');
videos=videos{:,1};

ff = fopen('/media/Drive2/sagar/EGTEA_Gaze_Plus/action_annotation/verb_idx.txt', 'r');
verb = textscan(ff, '%s %d32', 'Delimiter', '');
verb=verb{:,1};
[NoVe, ~]=size(verb);
Verbs={};
for i=1:NoVe
	temp=strsplit(verb{i,1});
	Verbs{i,1}=temp{1,1};
end

source='/media/Drive2/sagar/EGTEA_Gaze_Plus/pngs/';
[NoV,~]=size(videos);
cell_path={};
cell_label={};
count=0;
for i=1:NoV
	temp=videos{i,1};
	action=strsplit(temp);
	frames=dir(strcat(source,action{1,1},'/'));
	frames=frames(3:end,:);
	[NoF,~]=size(frames);
	for j=1:NoF
		count=count+1
		cell_path{count}=strcat(action{1,1},'/',frames(j).name);
		cell_label{count}=Verbs{str2num(action{1,3}),1};
	end
end
struct_CSV(1).Path=cell_path';
struct_CSV(1).Label=cell_label';
struct2csv(struct_CSV, 'train_label_EGtea.CSV');
fclose(ff);
fclose(f);

%% ---------------code for testing ------------------%%
clc;
clear;
f = fopen('/media/Drive2/sagar/EGTEA_Gaze_Plus/action_annotation/test_split1.txt', 'r');
videos = textscan(f, '%s %d32 %d32', 'Delimiter', '');
videos=videos{:,1};

ff = fopen('/media/Drive2/sagar/EGTEA_Gaze_Plus/action_annotation/verb_idx.txt', 'r');
verb = textscan(ff, '%s %d32', 'Delimiter', '');
verb=verb{:,1};
[NoVe, ~]=size(verb);
Verbs={};
for i=1:NoVe
	temp=strsplit(verb{i,1});
	Verbs{i,1}=temp{1,1};
end

source='/media/Drive2/sagar/EGTEA_Gaze_Plus/pngs/';
[NoV,~]=size(videos);
cell_path={};
cell_label={};
count=0;
for i=1:NoV
	temp=videos{i,1};
	action=strsplit(temp);
	frames=dir(strcat(source,action{1,1},'/'));
	frames=frames(3:end,:);
	[NoF,~]=size(frames);
	for j=1:NoF
		count=count+1
		cell_path{count}=strcat(action{1,1},'/',frames(j).name);
		cell_label{count}=Verbs{str2num(action{1,3}),1};
	end
end
struct_CSV(1).Path=cell_path';
struct_CSV(1).Label=cell_label';
struct2csv(struct_CSV, 'test_label_EGtea.CSV');
fclose(ff);
fclose(f);
