
function y = Fseriesval(a,b,t)
    N = length(b);
    ai = a(2:end);
    bi = b;

    % constant
    y = a(1)/2; 
    for k = 1:N
     y = y + ai(k)*cos(2*pi*k*t) + bi(k)*sin(2*pi*k*t) ;
    end
end