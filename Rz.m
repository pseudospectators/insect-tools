function Rz=Rz(angle)
    % rotation matrix around z axis
    Rz=[cos(angle) +sin(angle) 0;-sin(angle) cos(angle) 0; 0 0 1];
end
