olddir = pwd;
chdir('../faces');
directory_list = dir('*.pgm');
% get rid of . and ..
directory_list = directory_list(3:length(directory_list));
chdir(olddir);


load ../eyelocs.mat -ascii eyelocs

support = pgmRead('../support-trapezoid.pgm')>0;
