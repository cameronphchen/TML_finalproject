clear

dataname = 'ten_weeks'

load(['../data/raw/' dataname '.txt']);

data_raw = ten_weeks(:,3);

d_minute_size=length(data_raw);

d_halfhr_size=d_minute_size/30;
data = nan(d_halfhr_size,2);
num_hf_in_week = 7*24*2;

for i=1:d_halfhr_size
  tmp = mod(i,num_hf_in_week);
  if tmp~= 0
    data(i,1) = tmp;
  else
    data(i,1) = num_hf_in_week;
  end
  data(i,2) = mean(data_raw((i-1)*30+1:i*30),1);
end

training_data = data(1:size(data,1)/2,:);
testing_data = data(size(data,1)/2+1:size(data,1),:);

save(['../data/input/housepower_' dataname '_data'],'testing_data','training_data')

%data = household_power_consumption_refined
