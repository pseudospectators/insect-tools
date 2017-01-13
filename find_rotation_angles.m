% NOTE: below are testdriver scripts
function [alpha1, alpha2, alpha3] = find_rotation_angles( x1g, x2g, x3g, x1t, x2t, x3t, rotation_sequence )
    %% function [angles] = find_rotation_angles( x1g, x2g, x3g, x1t, x2t, x3t, rotation_sequence )
    %----------------------------------------------------------------------
    % given the set of vectors x1,x2,x3 (which should NOT be colinear) in
    % two coordinate systems, e.g. global (x1g,x2g,x3g) and transformed(
    % x1t,x2t,x3t), find the euler angles alpha1, alpha2, alpha3 that solve
    % the following eqn
    %
    % xt_i = Mx(alpha1) * My(alpha2) * Mz(alpha3) * xg_i  for i = 1,2,3
    %
    % Note that we assume the transformation to be pure rotation, no
    % translation is involved.
    %
    % The order of rotation matrices is passed in the string
    % rotation_sequence and can be
    %       xyz
    %       yxz
    %       zyx, ...
    %       flusi_wing (the same as )
    %       flusi_body (the same as xyz, the returned angles are then roll, pitch, yaw)
    %
    % This can for example be used to extract the yaw, pitch and roll angle
    % from digitized points using marker tracking. The tracking allows us
    % to obtain the body unit vectors in the global system (in the body
    % system, they are unit vectors, of course). This routine then finds
    % the corresponding angles.
    %
    %----------------------------------------------------------------------
   
    % intialize matlab's optimization toolbox
    opts = optimset('fsolve');
    % this gave the best results:
    opts.Algorithm = 'levenberg-marquardt';
    % start guess:
    x0 = [0;0;0];
    
    % it is tricky to pass additional arguments to functions, as a hack,
    % use global
    global params
    params.x1g = x1g;
    params.x2g = x2g;
    params.x3g = x3g;
    params.x1t = x1t;
    params.x2t = x2t;
    params.x3t = x3t;
    params.rotation_sequence = rotation_sequence;
    
    % solve the nonlinear optimization problem
    x = fsolve(@optimization_target_function, x0, opts);
    
    alpha1 = x(1);
    alpha2 = x(2);
    alpha3 = x(3);
end


function cost = optimization_target_function(x)
    %% cost function for the angles finder
    global params
    
    alpha1 = x(1);
    alpha2 = x(2);
    alpha3 = x(3);
    
    % construct the rotation matrix M as indicated by the string in the
    % parameters structure
    switch (params.rotation_sequence)
        case {'xyz','flusi_body','flusi-body'}
            M = Rx(alpha1)*Ry(alpha2)*Rz(alpha3);
        case {'yzx','flusi_wing','flusi-wing'}
            M = Ry(alpha1)*Rz(alpha2)*Rx(alpha3);
        otherwise
            error('unkown..')
    end
    
    % compute the result of the transformation using the currently iterated
    % angles. we know what result we would want (x1t etc), so this defines
    % our error or cost function directly
    x1t_computed = M * params.x1g;
    x2t_computed = M * params.x2g;
    x3t_computed = M * params.x3g;
    
    cost = abs( x1t_computed - params.x1t) +abs( x2t_computed - params.x2t) +abs( x3t_computed - params.x3t) ;
end


function testdriver_body2body
% a script to test the above functions using the body system back and forth
% (same angles are found as we put in)

    % given the body unit vectors (in the body system)
    x1b = [1; 0; 0];
    x2b = [0; 1; 0];
    x3b = [0; 0; 1];
    
    yaw = deg2rad( (rand(1)-0.5)*80);
    pitch = deg2rad(  (rand(1)-0.5)*80 );
    roll = deg2rad( (rand(1)-0.5)*80 );    
    M = Rx(roll)* Ry(pitch)* Rz(yaw);
    
    % bring these vectors to global system
    x1g = transpose(M) * x1b;
    x2g = transpose(M) * x2b;
    x3g = transpose(M) * x3b;

    % using the same convention, find the angles again from the
    % coordinates:
    [roll2, pitch2,yaw2] = find_rotation_angles( x1g, x2g, x3g, x1b, x2b, x3b, 'flusi-body' );
    
    % show result (works, is the same)
    fprintf('%f ', yaw, yaw2, pitch, pitch2, roll, roll2)
    fprintf('\n')
    
    M = transpose( Rx(roll2)* Ry(pitch2)* Rz(yaw2) ) ;
    % show errors (1e-13)
    x1g = M*x1b - x1g
    x2g = M*x2b - x2g
    x3g = M*x3b - x3g
end

function testdriver_body2wing
% a script to test the above functions using the body/wing system)
% (different angles are found as we put in, but same vectors)

    % given the body unit vectors (in the body system)
    x1b = [1; 0; 0];
    x2b = [0; 1; 0];
    x3b = [0; 0; 1];
    
    yaw = deg2rad( (rand(1)-0.5)*80);
    pitch = deg2rad(  (rand(1)-0.5)*80 );
    roll = deg2rad( (rand(1)-0.5)*80 );    
    M = Rx(roll)* Ry(pitch)* Rz(yaw);
    
    % bring these vectors to global system
    x1g = transpose(M) * x1b;
    x2g = transpose(M) * x2b;
    x3g = transpose(M) * x3b;

    % using the same convention, find the angles again from the
    % coordinates:
    [roll2, pitch2,yaw2] = find_rotation_angles( x1g, x2g, x3g, x1b, x2b, x3b, 'flusi-wing' );
    
    % show result (works, is the same)
    fprintf('%f ', yaw, yaw2, pitch, pitch2, roll, roll2)
    fprintf('\n')
    
    M = transpose( Ry(roll2)* Rz(pitch2)* Rx(yaw2) ) ;
    % show errors (1e-13)
    x1g = M*x1b - x1g
    x2g = M*x2b - x2g
    x3g = M*x3b - x3g
end