function value = read_str_inifile( file, section, keyword)
%% function value = read_str_inifile( file, section, keyword)
% from the specified ini file "file"
% find the section "section" (e.g. "[kinematics]" ) 
% and the keyword, e.g. variable=0.0202;
% and return the value, so '0.0202' as a string.
% 
% see also
% read_number_inifile



fid = fopen(file, 'r');
line='initializing';
section_found = 0;


while (ischar(line))
    line = fgetl(fid);
    
    % are we at the end of the file yet?
    if (ischar(line) )
        % trim leading and trailing spaces
        line_trim = strtrim(line);
        % is the line commented entirely?
        if ( numel(line_trim)>0 )
        if ((line_trim(1)~=';')&&(line_trim(1)~='#')&&(line_trim(1)~='%')&&(line_trim(1)~='!'))
            
            % remove comments from line:
            i2 = min([strfind(line_trim,'%'),strfind(line_trim,'#'),strfind(line_trim,';'),strfind(line_trim,'!')]);
            if ( isempty(i2) )
                line_nocomments = line_trim;
            else
                line_nocomments = line_trim( 1:i2-1 );
            end
            
            if (strfind(line_nocomments,['[' section ']']) )
                % we found the section!
                section_found = 1;
            end
            if (section_found==1) 
                % we found the section!
                if strfind(line_nocomments,keyword)
                    % we found the keyword!
                    value = line_nocomments( strfind(line_nocomments,'=')+1:end )  ;
                    if ( isempty(i2) )
                        warning('WARNING! line not properly terminated with colon ";"')
                    end                    
                    break
                end
            end
        end
        end
    else
        % end of file
        fprintf('file read but no value found!\n')
    end
end

end