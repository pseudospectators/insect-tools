function [a,b] = Fseries(t,y,n) % we assume here t=0...1

    if (size(t,1)==1)
        t=t';
    end

    if (size(y,1)==1)
        y=y';
    end
    
    t=2*pi*t;
    % make design matrix
    nx = t*(1:n);
    F = [0.5*ones(size(t)),cos(nx),sin(nx)];

    % Least squares
    c = F\y;

    % extract coefficients
    a = c(1:n+1);
    b = c(n+2:end);
end
