function d = stroke_avg_matrix(d1, tstroke)
%% function d = stroke_avg_matrix(d1, tstroke)
% Take the matrix d1(1:nt,1:N), which contains column-wise quantities from 
% a *.t file (forces, moments, etc..)
% and return the matrix of stroke averaged values of all columns

if (nargin==1)
    tstroke=1;
end

for i = 1:size(d1,2)
   [t,f] = stroke_avg_time_series( d1(:,1), d1(:,i), tstroke );
   d(:,1) = t;
   d(:,i) = f;
end

end