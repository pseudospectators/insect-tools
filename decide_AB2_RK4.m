d = load_t_file('dt.t');

dx = 1.25/640;
tmax = d(end,1);
nt=size(d,1);

eps = d(end,2)/d(end,5);

dt = d(:,2);
cfl = d(:,3);

umax = dx*cfl./dt;

fprintf('tmax=%f nt=%i cfl_mean=%f eps=%e dt_mean=%e\n',tmax,nt,mean(cfl),eps,tmax/nt)

%% simulate what would happen if u does not change
%% same level
CFL_RK4 = 2.0;
CFL_AB2 = 0.2;
eps_new = eps;%/4;
dx_new = dx;%/2;

fprintf('Assuming CFL_RK4=%f CFL_AB2=%f\n',CFL_RK4,CFL_AB2)

t_rk4 = 0;
it_rk4 = 0;
while t_rk4 < tmax
    u = interp1(d(:,1),umax,t_rk4,'nearest');
    dt_rk4 = min( [ CFL_RK4*dx_new/u, 0.99*eps_new ] );    
    t_rk4 = t_rk4 + dt_rk4;
    it_rk4 = it_rk4 + 1;
end

t_ab2 = 0;
it_ab2 = 0;
while t_ab2<tmax
    u = interp1(d(:,1),umax,t_ab2,'nearest');
    dt_ab2 = min( [ CFL_AB2*dx_new/u, 0.99*eps_new ] );    
    t_ab2 = t_ab2 + dt_ab2;
    it_ab2 = it_ab2 + 1;
end

fprintf('---------SAME LEVEL--------------------\n');
fprintf('AB2: nt=%i nrhs=%i dt_mean=%e\n',it_ab2, it_ab2, tmax/it_ab2)
fprintf('RK4: nt=%i nrhs=%i dt_mean=%e\n',it_rk4, it_rk4*4, tmax/it_rk4)
if ( it_rk4*4 > it_ab2 )
    fprintf('AB2 is faster (factor=%f)\n',it_rk4*4/it_ab2)
else
    fprintf('RK4 is faster (factor=%f)\n',it_ab2/it_rk4/4)
end

%% one level up
CFL_RK4 = 2.0;
CFL_AB2 = 0.2;
eps_new = eps/4;
dx_new = dx/2;

t_rk4 = 0;
it_rk4 = 0;
while t_rk4 < tmax
    u = interp1(d(:,1),umax,t_rk4,'nearest');
    dt_rk4 = min( [ CFL_RK4*dx_new/u, 0.99*eps_new ] );    
    t_rk4 = t_rk4 + dt_rk4;
    it_rk4 = it_rk4 + 1;
end

t_ab2 = 0;
it_ab2 = 0;
while t_ab2<tmax
    u = interp1(d(:,1),umax,t_ab2,'nearest');
    dt_ab2 = min( [ CFL_AB2*dx_new/u, 0.99*eps_new ] );    
    t_ab2 = t_ab2 + dt_ab2;
    it_ab2 = it_ab2 + 1;
end

fprintf('---------ONE LEVEL UP--------------------\n');
fprintf('AB2: nt=%i nrhs=%i dt_mean=%e\n',it_ab2, it_ab2, tmax/it_ab2)
fprintf('RK4: nt=%i nrhs=%i dt_mean=%e\n',it_rk4, it_rk4*4, tmax/it_rk4)
if ( it_rk4*4 > it_ab2 )
    fprintf('AB2 is faster (factor=%f)\n',it_rk4*4/it_ab2)
else
    fprintf('RK4 is faster (factor=%f)\n',it_ab2/it_rk4/4)
end

%% two levels up
CFL_RK4 = 2.0;
CFL_AB2 = 0.2;
eps_new = eps/4/4;
dx_new = dx/2/2;

t_rk4 = 0;
it_rk4 = 0;
while t_rk4 < tmax
    u = interp1(d(:,1),umax,t_rk4,'nearest');
    dt_rk4 = min( [ CFL_RK4*dx_new/u, 0.99*eps_new ] );    
    t_rk4 = t_rk4 + dt_rk4;
    it_rk4 = it_rk4 + 1;
end

t_ab2 = 0;
it_ab2 = 0;
while t_ab2<tmax
    u = interp1(d(:,1),umax,t_ab2,'nearest');
    dt_ab2 = min( [ CFL_AB2*dx_new/u, 0.99*eps_new ] );    
    t_ab2 = t_ab2 + dt_ab2;
    it_ab2 = it_ab2 + 1;
end

fprintf('---------TWO LEVEL UP--------------------\n');
fprintf('AB2: nt=%i nrhs=%i dt_mean=%e\n',it_ab2, it_ab2, tmax/it_ab2)
fprintf('RK4: nt=%i nrhs=%i dt_mean=%e\n',it_rk4, it_rk4*4, tmax/it_rk4)
if ( it_rk4*4 > it_ab2 )
    fprintf('AB2 is faster (factor=%f)\n',it_rk4*4/it_ab2)
else
    fprintf('RK4 is faster (factor=%f)\n',it_ab2/it_rk4/4)
end
fprintf('-----------------------------------------\n');