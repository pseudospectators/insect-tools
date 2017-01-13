function myfig(ifig,AR,factor,font)
%% new_myfig(ifig)
% creates a new figure of type myfig

    if (nargin == 0)
        h_fig = figure;
    else
        h_fig = figure(ifig);
    end
    
    if (nargin == 1)
        % default aspect ratio
        AR = 6.5/20.4;
        factor=1.2;
        font=9;
    elseif (nargin ==2)
        factor=1.2;
        font=9;
    end
        

%     clf; 

    set(h_fig,'DefaultaxesFontSize',font);
    set(h_fig,'DefaulttextFontsize',font);
    set(h_fig,'DefaultaxesFontName','Times');
    set(h_fig,'DefaulttextFontName','Times');


    set(h_fig,'Units','centimeters','Position',factor*[2 5 20.4 AR*20.4],'Resize','on','PaperPositionMode','auto');

    hold on;
    box on
end
