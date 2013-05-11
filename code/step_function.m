% step function
% [s] = step_function(location, type, datalength)
% location: place the value change
% type: 0:all 0 1:0->1, 2:0->-1, 3:1->0, 4:-1->0
% datalength: size of the step function
%
% by Cameron P.H. Chen @Princeton 
% contact: pohsuan [at] princeton [dot] edu



function [s] = step_function(location, type,d_range)
s = zeros(d_range,length(location));


%assert( (max(type)<=4)&&(min(type)>=0), 'type shold be <=4, and <=0');
%assert( length(location)==length(type), ...
%        'location length and type length should be the same');
for i = 1:length(location)
  switch type(i) 
    case 1
      s(location(i):end,i)=1;
    case 2
      s(location(i):end,i)=-1;
    case 3
      s(1:location(i),i)=1;
    case 4
      s(1:location(i),i)=-1;
    otherwise
      ;
  end
end



