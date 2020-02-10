%  create a channel map file

Nchannels = 32;
connected = true(Nchannels, 1);
chanMap   = 1:Nchannels;
chanMap0ind = chanMap - 1;
xcoords   = ones(Nchannels,1);
ycoords   = [1:Nchannels]';
kcoords   = ones(Nchannels,1); % grouping of channels (i.e. tetrode groups)

fs = 25000; % sampling frequency
save('C:\DATA\Spikes\20150601_chan32_4_900s\chanMap.mat', ...
    'chanMap','connected', 'xcoords', 'ycoords', 'kcoords', 'chanMap0ind', 'fs')

%%

Nchannels = 32;
connected = true(Nchannels, 1);
chanMap   = 1:Nchannels;
chanMap0ind = chanMap - 1;

xcoords   = repmat([1 2 3 4]', 1, Nchannels/4);
xcoords   = xcoords(:);
ycoords   = repmat(1:Nchannels/4, 4, 1);
ycoords   = ycoords(:);
kcoords   = ones(Nchannels,1); % grouping of channels (i.e. tetrode groups)

fs = 25000; % sampling frequency

save('C:\DATA\Spikes\Piroska\chanMap.mat', ...
    'chanMap','connected', 'xcoords', 'ycoords', 'kcoords', 'chanMap0ind', 'fs')
%%

% kcoords is used to forcefully restrict templates to channels in the same
% channel group. An option can be set in the master_file to allow a fraction 
% of all templates to span more channel groups, so that they can capture shared 
% noise across all channels. This option is

% ops.criterionNoiseChannels = 0.2; 

% if this number is less than 1, it will be treated as a fraction of the total number of clusters

% if this number is larger than 1, it will be treated as the "effective
% number" of channel groups at which to set the threshold. So if a template
% occupies more than this many channel groups, it will not be restricted to
% a single channel group. 


Nchannels = 4;
connected = true(Nchannels, 1);
chanMap   = 1:Nchannels;
chanMap0ind = chanMap - 1;
xcoords   = [0 -20 20 0]';
ycoords   = [0 20 20 40]';
kcoords   = ones(Nchannels,1); % grouping of channels (i.e. tetrode groups)

fs = 30000; % sampling frequency
save('/auto/data/code/KiloSort/chanMap_Thomas_Tetrode.mat', ...
    'chanMap','connected', 'xcoords', 'ycoords', 'kcoords', 'chanMap0ind', 'fs')
%%
[s,UCLA_to_OEP]=probe_128D();
x=s.x(UCLA_to_OEP);
round_factor=100;
x=round(x/round_factor)*round_factor;
[unx,~,uni]=unique(x);
Nchannels = 128;
connected = true(Nchannels, 1);
chanMap   = 1:Nchannels;
chanMap0ind = chanMap - 1;
xcoords   = s.x(UCLA_to_OEP);%[0 -20 20 0]';
ycoords   = s.z(UCLA_to_OEP);%[0 20 20 40]';
kcoords   = uni; % grouping of channels (i.e. tetrode groups)
chanMap0ind = chanMap - 1;
fs = 30000; % sampling frequency
save('/auto/data/code/KiloSort/chanMap_128D.mat', ...
    'chanMap','connected', 'xcoords', 'ycoords', 'kcoords', 'chanMap0ind', 'fs')

%%
[s,UCLA_to_OEP]=probe_64D();
x=s.x(UCLA_to_OEP);
round_factor=100;
x=round(x/round_factor)*round_factor;
[unx,~,uni]=unique(x);
Nchannels = 64;
connected = true(Nchannels, 1);
chanMap   = UCLA_to_OEP+64;
chanMap0ind = chanMap - 1;
xcoords   = s.x(UCLA_to_OEP);%[0 -20 20 0]';
ycoords   = s.z(UCLA_to_OEP);%[0 20 20 40]';
kcoords   = uni; % grouping of channels (i.e. tetrode groups)
chanMap0ind = chanMap - 1;
fs = 30000; % sampling frequency
save('/auto/data/code/KiloSort/chanMap_64D_slot2.mat', ...
    'chanMap','connected', 'xcoords', 'ycoords', 'kcoords', 'chanMap0ind', 'fs')
%%
% 64D slot 1 bottom 
[s,UCLA_to_OEP]=probe_64D_bottom();
x=s.x(UCLA_to_OEP);
round_factor=100;
x=round(x/round_factor)*round_factor;
[unx,~,uni]=unique(x);
Nchannels = 64;
connected = true(Nchannels, 1);
chanMap   = UCLA_to_OEP;
chanMap0ind = chanMap - 1;
xcoords   = s.x(UCLA_to_OEP);%[0 -20 20 0]';
ycoords   = s.z(UCLA_to_OEP);%[0 20 20 40]';
kcoords   = uni; % grouping of channels (i.e. tetrode groups)
chanMap0ind = chanMap - 1;
fs = 30000; % sampling frequency
save('/auto/data/code/KiloSort/chanMap_64D_slot1_bottom.mat', ...
    'chanMap','connected', 'xcoords', 'ycoords', 'kcoords', 'chanMap0ind', 'fs')

%%
% 64D slot 2 bottom 
[s,UCLA_to_OEP]=probe_64D_bottom();
x=s.x(UCLA_to_OEP);
round_factor=100;
x=round(x/round_factor)*round_factor;
[unx,~,uni]=unique(x);
Nchannels = 64;
connected = true(Nchannels, 1);
chanMap   = UCLA_to_OEP+64;
chanMap0ind = chanMap - 1;
xcoords   = s.x(UCLA_to_OEP);%[0 -20 20 0]';
ycoords   = s.z(UCLA_to_OEP);%[0 20 20 40]';
kcoords   = uni; % grouping of channels (i.e. tetrode groups)
chanMap0ind = chanMap - 1;
fs = 30000; % sampling frequency
save('/auto/data/code/KiloSort/chanMap_64D_slot2_bottom.mat', ...
    'chanMap','connected', 'xcoords', 'ycoords', 'kcoords', 'chanMap0ind', 'fs')