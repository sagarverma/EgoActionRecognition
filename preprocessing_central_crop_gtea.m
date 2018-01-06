clc;
clear;
%---------code to generate CSV file and putting .png into class folders for
%RGB dataset using mylabes text files.
clc;
clear;
destination='/home/shubham/Egocentric/dataset/GTea_preprocessed_rgb_11_actions_ctrl_crop_280_450/train/';
actions=dir(destination);
actions=actions(3:end);
path_sub='/home/shubham/Egocentric/dataset/GTea/png/';
subjects=dir(path_sub);
path_label='/home/shubham/Egocentric/dataset/GTea/gtea_labels_mylabels/';
labels=dir(path_label);
subjects=subjects(3:end);
labels=labels(3:end);
[total_sub,~]=size(subjects);
count=1;
cell_temp_sub_path={};
cell_label={};
%struct_CSV = struct('Name',{},'Path',{},'Label',{});
for i=1:total_sub-7
    i
    label_txt=strcat(labels(i).folder,'/',labels(i).name)
    subject_path=strcat(subjects(i).folder,'/',subjects(i).name,'/')
    fid   = fopen(label_txt);
    tline = fgetl(fid);
    count
    while ischar(tline)
        temp=tline;
        line=strsplit(temp);
        [~,yolo]=size(line);
        if yolo>3
            action=string(line{1,1});
            if action==string('x')
                action=string('bg');
            end
%             if action==string('open')
%                 action=string('close');
%             end
            from=str2double(line(end-2));
            to=str2double(line(end-1));
            for j=from:to
                image_path=strcat(subject_path,num2str(j,'%.10d'),'.png');
                temp_subject_path=strcat(subjects(i).folder,'/');
                subject_frame=strcat(subjects(i).name,'/',num2str(j,'%.10d'),'.png');
                temp_image=imread(image_path);
                [hig wid,~]=size(temp_image);
                cntr_x=int16(wid/2); cntr_y=int16(hig/2);
                %temp_image=temp_image(cntr_y-199:cntr_y+199+1, cntr_x-199:cntr_x+199+1,:);
                temp_image=temp_image(cntr_y-140:cntr_y+139, cntr_x-225:cntr_x+224,:);
                %imshowpair(temp_image, imread(image_path),'montage');
                image_destination=(strcat(destination, char(action),'/',num2str(count,'%.10d'),'.png'));
                imwrite(temp_image, image_destination);
                cell_temp_sub_path{count}=image_destination;
                cell_label{count}=char(action);
                count=count+1
%                 if count==320
%                     imwrite(temp_image,'320_AR.png')
%                 end
            end
        end    
        tline = fgetl(fid);
    end
    fclose(fid);
end
struct_CSV(1).Path=cell_temp_sub_path';
struct_CSV(1).Label=cell_label';
struct2csv(struct_CSV, 'train_label_mylabels_11_classes_rgb_central_crop_280_450.CSV');
% 
% % %%%%%%%%%%%%%%% ----------------for validation
% % %%%%%%%%%%%%%%% ----------------------------__%%%%%%%%%%%%
destination='/home/shubham/Egocentric/dataset/GTea_preprocessed_rgb_11_actions_ctrl_crop_280_450/val/';
count=1;
cell_temp_sub_path={};
cell_label={};
%struct_CSV = struct('Name',{},'Path',{},'Label',{});
for i=22:total_sub
    i
    label_txt=strcat(labels(i).folder,'/',labels(i).name)
    subject_path=strcat(subjects(i).folder,'/',subjects(i).name,'/')
    fid   = fopen(label_txt);
    tline = fgetl(fid);
    while ischar(tline)
        temp=tline;
        line=strsplit(temp);
        [~,yolo]=size(line);
        if yolo>3
            action=string(line{1,1});
            if action==string('x')
                action=string('bg');
            end
%             if action==string('open')
%                 action=string('close');
%             end
            from=str2double(line(end-2));
            to=str2double(line(end-1));             
            for j=from:to
                image_path=strcat(subject_path,num2str(j,'%.10d'),'.png');
                temp_subject_path=strcat(subjects(i).folder,'/');
                subject_frame=strcat(subjects(i).name,'/',num2str(j,'%.10d'),'.png');
                temp_image=imread(image_path);
                [hig wid,~]=size(temp_image);
                cntr_x=int16(wid/2); cntr_y=int16(hig/2);
                temp_image=temp_image(cntr_y-199:cntr_y+199+1, cntr_x-199:cntr_x+199+1,:);
                image_destination=(strcat(destination,char(action),'/',num2str(count,'%.10d'),'.png'));
                imwrite(temp_image, image_destination);
                cell_temp_sub_path{count}=image_destination;
                cell_label{count}=char(action);
                count=count+1;
               
            end
        end    
        tline = fgetl(fid);
    end
    fclose(fid);
  
end
% 
struct_CSV(1).Path=cell_temp_sub_path';
struct_CSV(1).Label=cell_label';
struct2csv(struct_CSV, 'validation_label_mylabels_11_classes_rgb_central_crop_280_450.CSV');

%% code to resize bg class only -------------------%%
% temp=dir('/home/shubham/Egocentric/dataset/bg_modified');
% destination='/home/shubham/Egocentric/dataset/GTea_preprocessed_matlab/train/';
% temp=temp(3:end);
% [total,~]=size(temp);
% count=1;
% for i=1:total
%     img_path=strcat(temp(i).folder,'/',temp(i).name)
%     temp_image=imread(img_path);
%     
%     temp_image=imresize(temp_image,[227,227]);
%     image_destination=(strcat(destination,'bg','/',num2str(count,'%.10d'),'.png'));
%     imwrite(temp_image,image_destination);
%     count=count+1;
% end
