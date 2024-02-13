prompt = {'Enter number of matrices:','Enter row size:','Enter column size:','Enter density:','Matrix ill-conditioned? (True/False)'};
dlgtitle = 'Define random rectangular matrices';
dims = [1 35];
definput = {'5','100','50','1','False'};
answer = inputdlg(prompt,dlgtitle,dims,definput);
maxIter = str2num(answer{1});
m = str2double(answer{2});
n = str2double(answer{3});
density = str2double(answer{4});
isIll = answer{5};

if m == n 
    uiwait(warndlg('Error: matrix is not rectangular.'));
elseif m <= 0 || n <= 0
    uiwait(warndlg('Error: size must be greater than 0.'));
elseif strcmp(isIll,'True') == 0 && strcmp(isIll,'False') == 0
    uiwait(warndlg('Error: set True or False value.'));
elseif density <= 0
    uiwait(warndlg('Error: density must be greater than 0.'));
else
    homeDir = ('matrices');
    if (~exist(homeDir, 'dir')); mkdir(homeDir); end
    
    for i = 1:maxIter
        distribution = randi(150);
        if strcmp(isIll,'True') == 1
            r = hilb(m);
        else
            if density == 1
                A = rand(m,n);
                r = -distribution + (distribution*2).*A;
            else
                A = sprand(m, n, density);
                A = (distribution*2)*A - distribution*spones(A);
                r = full(A);
            end
        end
    
        filename = strcat(homeDir, '/', 'matrix', num2str(m), 'x', num2str(n), '_', num2str(i), '.txt');
        dlmwrite(filename, r, 'delimiter', '\t', 'precision',3)
    end
end