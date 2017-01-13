function myfig_print(name,ifig)
    %% myfig_print(name,[ifig])
    % print figure to eps file
    
    if (nargin == 1)
        h_fig = gcf;
    else
        h_fig = figure(ifig);
    end
    
    print(h_fig,'-depsc',name);

end