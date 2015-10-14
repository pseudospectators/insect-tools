function [tavg,favg]= stroke_avg_time_series( t, f, tstroke )
%% stroke_avg_time_series( t, f, tstroke )
% take the signal f(t) and return the stroke-averages for each stroke 
% the signal f(t) does not need to be equidistant
% tstroke is optional (default=1)

if (nargin==2)
    tstroke=1;
end

tmax = t(end);
n = ceil(tmax/tstroke);

% check if we're not actually fairly close to having the next period
% complete, then we can extrapolate some reasonable value

if ( n*tstroke - tmax) >0.025
    % it is not complete, decrease
    n = n-1;
end

% fprintf('we have %i complete strokes\n', n)

tavg=zeros(n,1);
favg=zeros(n,1);

if (length(t)~=length(f))
    warning('length are not equal')
    return
end

for i = 1:n
   t1 = (i-1)*tstroke;
   t2 = (i)*tstroke;
   
   tavg(i) = 0.5*(t1+t2);
   favg(i) = stroke_avg(t,f,t1,t2,'linear');
end
