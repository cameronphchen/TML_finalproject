% run experiment for sparsity gradient boosing & 
% standard gradient boosting
% by Cameron P.H. Chen @Princeton 
% contact: pohsuan [at] princeton [dot] edu

clear
close all

% set the algorithm options
fprintf('loading options\n')
options.data_name = 'housepower_twenty_weeks';
options.time = clock;
options.time = [date '-' num2str(options.time(4)) num2str(options.time(5))];
options.input_path = '../data/input/'; 
options.working_path = '../data/working/' ; 
options.output_path = '../data/output/' ; 
options.random_seed = 1;
options.exp_to_run = {  'GB', 'SGB'}; %Gradient Boosting,Sparse GB
options.hypothesis_space = 'step_function';
options.cross_valid = 2;
options.num_iter = 50;
options.num_testing_week = 5;

parameters.sparsity_constarint = 200;


% load the data
fprintf('loading data\n')
load([ options.input_path options.data_name '_data.mat']);

testing_data_raw= testing_data;
testing_data = testing_data(1:options.num_testing_week*336,:);


d_size = size(training_data,1);
d_min  = min(training_data(:,1));
d_max  = max(training_data(:,1));
d_range = d_max -d_min +1;
 
%testing_week_num = size(testing_data,1)/336;
training_week_num = size(training_data,1)/336; 

fprintf('gradient boosting\n')
[alpha_gb r_train_gb F_gb] = gradient_boosting(training_data,...
                                              options.num_iter,d_range);
fprintf('sparse gradient boosting\n')

[alpha_sgb r_train_sgb F_sgb] = sparse_gradient_boosting(training_data,...
                    options.num_iter,d_range,parameters.sparsity_constarint);


%calculate gb training error
training_loss_gb = zeros(options.num_iter+1,1);
for i=1:options.num_iter+1
  tmp = 0; 
  for k=1:size(training_data,1)
    tmp = tmp + (training_data(k,2) - F_gb(training_data(k,1),i))^2;
  end 
    training_loss_gb(i,1) = tmp/size(training_data,1);
end


%calculate sgb training error
training_loss_sgb = zeros(options.num_iter+1,1);
for i=1:options.num_iter+1
  tmp = 0; 
  for k=1:size(training_data,1)
    tmp = tmp + (training_data(k,2) - F_sgb(training_data(k,1),i))^2;
  end 
    training_loss_sgb(i,1) = tmp/size(training_data,1);
end



%calculate gb testing error
testing_loss_gb = zeros(options.num_iter+1,1);
for i=1:options.num_iter+1
  tmp = 0; 
  for k=1:size(testing_data,1)
    tmp = tmp + (testing_data(k,2) - F_gb(testing_data(k,1),i))^2;
  end 
    testing_loss_gb(i,1) = tmp/size(testing_data,1);
end

%calculate sgb testing error
testing_loss_sgb = zeros(options.num_iter+1,1);
for i=1:options.num_iter+1
  tmp = 0; 
  for k=1:size(testing_data,1)
    tmp = tmp + (testing_data(k,2) - F_sgb(testing_data(k,1),i))^2;
  end 
    testing_loss_sgb(i,1) = tmp/size(testing_data,1);
end


% plot iteration vs training/testing error
figure
hold on
grid on
plot(1:options.num_iter+1,training_loss_gb,'b*-','Linewidth',2)
plot(1:options.num_iter+1,testing_loss_gb,'r*-','Linewidth',2)
plot(1:options.num_iter+1,training_loss_sgb,'g','Linewidth',2)
plot(1:options.num_iter+1,testing_loss_sgb,'k','Linewidth',2)
legend('GB training error','GB testing error',...
       'SGB training error','SGB testing error')
hold off
saveas(gcf,[options.output_path '/' 'err_' options.data_name '_' options.num_iter], 'tiff')


% plot iteration vs number of hypothesis selected
figure
hold on
grid on
sparsity_gb(1,:) = sum(sum(alpha_gb~=0))
sparsity_sgb(1,:) = sum(sum(alpha_sgb~=0))
plot(1:options.num_iter+1, sparsity_gb,'b*-','Linewidth',2)
plot(1:options.num_iter+1, sparsity_sgb,'r','Linewidth',2)
legend('GB sparsity','SGB sparsity')
hold off
saveas(gcf,  [ options.output_path '/' 'sparsity_' options.data_name '_' num2str(options.num_iter)], 'tiff')






% plot training regression result
gif_filename =[ options.output_path '/' options.data_name '_' num2str(options.num_iter) '.gif' ] ;




%{
set(gca,'nextplot','replacechildren','visible','off')
for i=1:options.num_iter
  hold on
  grid on
  scatter(training_data(:,1),training_data(:,2),100*ones(size(training_data,1),1),'.');
  plot(1:size(F_gb,1),F_gb(:,i),'r','Linewidth',2);
  f = getframe;
  im = frame2im(f);
  [imind,cm] = rgb2ind(im,256);
  if i == 1;
    imwrite(imind,cm,gif_filename,'gif', 'Loopcount',inf);
  else
    imwrite(imind,cm,gif_filename,'gif','WriteMode','append');
  end
  hold off
  close all
  %saveas(gcf,  [ options.output_path '/' options.data_name '_' options.num_iter], 'tiff')
end
%}
