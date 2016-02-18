function visualize_wingpath_chord( file, varargin )
%% function visualize_wingpath_chord( file, varargin )
% read a kinematics.in file, and draw the wingtip path with a chord line
% that illustrates the angle of attack
% -------------------------------------------------------------------------
% Input:
%   file: the kinematics.in file to be visualized
%   'beta': body pitch angle (in degree)
%   'gamma': body yaw angle (in degree)
%   'psi': body roll angle (in degree)
%   'eta_stroke': anatomical stroke plane angle (in degree)
%   'draw_path' : draw the path or not?
%   'pivot': pivot point coordinates in body system (LEFT WING)
% -------------------------------------------------------------------------
% NOTE: This concerns the LEFT wing in parameter file
% -------------------------------------------------------------------------
%   Example call:
%
% visualize_wingpath_chord( file, 'beta',20,'gamma',0,'psi',0,'eta_stroke',37.5 )
%
% Arguments do not have to be in the right order
% If Missing angle, 0 is assumed.
% -------------------------------------------------------------------------

% read in the kinematics file
kine = read_kinematics_file( file );  

%% default values for body angles:
psi = 0;
beta = 0;
gamma = 0;
eta_stroke = 0;
draw_path=0;
pivot=[0.3 ;+0.23 ;0.23];

%% read in optional arguments
i = 1;
while i <= length(varargin)
   switch varargin{i}
       case 'psi'
           % check if we're not at the end of the parameter list
           assert(i+1<=numel(varargin),'missing an argument!');
           i = i+1;
           % check if next argument is numeric
           if (~isnumeric(varargin{i}))
               error(['Parameter value for ' varargin{i-1} ' must be numeric!']);
           end
           % take that value
           psi = varargin{i};
       case 'beta'
           assert(i+1<=numel(varargin),'missing an argument!');
           i = i+1;
           if (~isnumeric(varargin{i}))
               error(['Parameter value for ' varargin{i-1} ' must be numeric!']);
           end
           % take that value
           beta = varargin{i};
       case 'gamma'
           assert(i+1<=numel(varargin),'missing an argument!');
           i = i+1;
           if (~isnumeric(varargin{i}))
               error(['Parameter value for ' varargin{i-1} ' must be numeric!']);
           end
           % take that value
           gamma = varargin{i};
       case 'eta_stroke'
           assert(i+1<=numel(varargin),'missing an argument!');
           i = i+1;
           if (~isnumeric(varargin{i}))
               error(['Parameter value for ' varargin{i-1} ' must be numeric!']);
           end
           % take that value
           eta_stroke = varargin{i};
       case 'draw_path'
           assert(i+1<=numel(varargin),'missing an argument!');
           i = i+1;
           if (~isnumeric(varargin{i}))
               error(['Parameter value for ' varargin{i-1} ' must be numeric!']);
           end
           % take that value
           draw_path = varargin{i};
       case 'pivot'
           assert(i+1<=numel(varargin),'missing an argument!');
           i = i+1;
           if (~isnumeric(varargin{i}))
               error(['Parameter value for ' varargin{i-1} ' must be numeric!']);
           end
           % take that value
           pivot = varargin{i};
       otherwise
           error(['unkown parameter:' varargin{i}])
   end
   i=i+1;
end

fprintf('Draw wingpath with psi_roll=%2.1f beta_pitch=%2.1f gamma_yaw=%2.1f eta_stroke=%2.1f pivot=[%2.1f %2.1f %2.1f]\n',...
        psi,beta,gamma,eta_stroke,pivot)

%----------------------------------------------------------------------------------------
%% construct angles from Fourier series
time = 0:2.5e-2:1;
[phi,alpha,theta] = evaluate_kinematics_file_time(time,kine);

% body angles:
psi   =  psi*pi/180;
beta  =  beta*pi/180;
gamma =  gamma*pi/180;
% stroke plane angle (anatomical, i.e. relative to body)
eta_stroke = eta_stroke*pi/180;

% wing tip point in wing coordinate system:
x_wing = [0;1;0];

% rotation matrix from global to body coordinate system:
M_body=Rx(psi)*Ry(beta)*Rz(gamma);


% rotation matrix from body to stroke coordinate system:
M_stroke_l = Ry(eta_stroke);

% plot properties
wing_chord = 0.1;
circle_radius = wing_chord/6;


% coordinate of pivot point (in body coordinate system!)
x_pivot = pivot;
% we assume the body to be at the origin:
xc_body = [0 0 0]';



%% PLOT
% Create figure
h_fig = figure;
set(h_fig,'Units','centimeters','Position',0.65*[10 8.0 14.0 8.8],'Resize','on','PaperPositionMode','auto'); 
clf;
box on
hold on

set(0,'DefaultaxesFontSize',9);
set(0,'DefaulttextFontsize',9);
set(0,'DefaultaxesFontName','Times');
set(0,'DefaulttextFontName','Times');

% get colors
C = colormap(jet(length(time)));

%% draw chords
for it=1:length(time)
    % get current angles in radiants
    alpha_l = alpha(it)*(pi/180);
    theta_l = theta(it)*(pi/180);
    phi_l   = phi(it)  *(pi/180);
    
    % rotation matrix from body to wing coordinate system:
    M_wing_l = Ry(alpha_l)*Rz(theta_l)*Rx(phi_l)*M_stroke_l;
    
    % these are in the body coordinate system:
    xc   = transpose(M_body) * (transpose(M_wing_l)*x_wing  + x_pivot );
    x_le = transpose(M_body) * (transpose(M_wing_l)*[ 0.5*wing_chord;1;0] + x_pivot);
    x_te = transpose(M_body) * (transpose(M_wing_l)*[-0.5*wing_chord;1;0] + x_pivot);
    
    % color all cuts differently
    color = C(it,:);
   
    % uncomment the following if you want to mark the wing tip coordinate:    
    % plot( xc(1),xc(3),'*','color',color)
    
    % unit vector from leading to trailing edge    
    factor=wing_chord / sqrt( (x_le(1)-x_te(1))^2 + (x_le(3)-x_te(3))^2 );
    e1 = (x_te(1)-x_le(1)) * factor;
    e3 = (x_te(3)-x_le(3)) * factor;
    line( [x_le(1) x_le(1)+e1], [x_le(3) x_le(3)+e3] , 'color',color)    
    rectangle('Position',[x_le(1)-circle_radius/2 x_le(3)-circle_radius/2, circle_radius, circle_radius],...
              'Curvature',[1 1],'FaceColor',color,'EdgeColor',color);
end

%% mark pivot point

x_pivot_glob = transpose(M_body) * x_pivot;
plot( x_pivot_glob(1), x_pivot_glob(3),'k+')


%% draw stroke plane dashed line
M_stroke = Ry(eta_stroke);
% we draw the line between [0,0,-1] and [0,0,1] in the stroke system
xs1 = [0 0 1]'; xs2=[0 0 -1]';
% bring these points back to the global system
x1 = transpose(M_body) * ( transpose(M_stroke)*xs1 + x_pivot );
x2 = transpose(M_body) * ( transpose(M_stroke)*xs2 + x_pivot );
% remember we're in the x-z plane
line ( [x1(1) x2(1)], [x1(3) x2(3)], 'color','k','linestyle','--' )

  
%% draw the path line, if desired  
if (draw_path==1)
    % create fine-sampled time vector
    time = 0:1e-3:1;
    % and evaluate the kinematics file on it
    [phi,alpha,theta] = evaluate_kinematics_file_time(time,kine);
    
    for it=1:length(time)
        alpha_l = alpha(it)*(pi/180);
        theta_l = theta(it)*(pi/180);
        phi_l   = phi(it)  *(pi/180);
        
        % rotation matrix from body to wing coordinate system:
        M_wing_l = Ry(alpha_l)*Rz(theta_l)*Rx(phi_l)*M_stroke_l;
        
        % these are in the body coordinate system:                
        xc(:,it) = transpose(M_body) * (transpose(M_wing_l) * x_wing  + x_pivot) ;
    end
    plot(xc(1,:),xc(3,:),'k')
end
  
xlabel('x (global)');
ylabel('z (global)');
axis equal



% Print figure
print('-depsc', [file '_path.eps']);

end
