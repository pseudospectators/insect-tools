function value = input_parameter_from_varargin( name, default, varargin )
    %% function input_parameter_from_varargin( name, value, default )
    % in a cell structure generally labeled varargin, look for a value
    % and if not found, assign the default value.
    % 
    % many matlab routines are called like
    %   a = testfunction(b,c,'degrees','yes','time',0.9)
    % and they can be defined as
    %   function a = testfunction(b,c,varargin)
    % then the testfunction can call 
    %   time =  input_parameter_from_varargin( varargin, 'time', 0.0 )
    % which returns 0.9 in this case. if the value 'time' is not found, 0.0
    % is returned.

    % first assign default value
    value = default;

    for i = 1:length(varargin{end})-1
        if strcmp(varargin{end}{i}, name)
            value = varargin{end}{i+1};
            break
        end
    end
end
