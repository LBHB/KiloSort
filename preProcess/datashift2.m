function rez = datashift2(rez, do_correction)

NrankPC = 6;
[wTEMP, wPCA]    = extractTemplatesfromSnippets(rez, NrankPC);
rez.wTEMP = gather(wTEMP);
rez.wPCA  = gather(wPCA);

ops = rez.ops;

% The min and max of the y and x ranges of the channels
ymin = min(rez.yc);
ymax = max(rez.yc);
xmin = min(rez.xc);
xmax = max(rez.xc);

dmin = median(diff(unique(rez.yc)));
fprintf('vertical pitch size is %d \n', dmin)
rez.ops.dmin = dmin;
rez.ops.yup = ymin:dmin/2:ymax; % centers of the upsampled y positions

% dminx = median(diff(unique(rez.xc)));
yunq = unique(rez.yc);
mxc = zeros(numel(yunq)-1, 1);
num_E_per_yunq=arrayfun(@(y)length(rez.xc(rez.yc==y)),yunq);
if any(num_E_per_yunq==1)
    do_two_rows_at_a_time=true;
    inds=1:numel(yunq)-1;
else
    do_two_rows_at_a_time=false;
    inds=1:numel(yunq);
end
for j = inds
    if do_two_rows_at_a_time
        xc = rez.xc(rez.yc==yunq(j) | rez.yc==yunq(j+1));
    else
        xc = rez.xc(rez.yc==yunq(j));
    end
    if numel(xc)>1
       mxc(j) = median(diff(sort(xc))); 
    end
end
dminx = max(5, median(mxc));
fprintf('horizontal pitch size is %d \n', dminx)

rez.ops.dminx = dminx;
nx = round((xmax-xmin) / (dminx/2)) + 1;
rez.ops.xup = linspace(xmin, xmax, nx); % centers of the upsampled x positions
disp(rez.ops.xup) 


if  getOr(rez.ops, 'nblocks', 1)==0
    rez.iorig = 1:rez.temp.Nbatch;
    return;
end



% binning width across Y (um)
dd = 5;
% min and max for the range of depths
dmin = ymin - 1;
dmax  = 1 + ceil((ymax-dmin)/dd);
disp(dmax)


spkTh = 10; % same as the usual "template amplitude", but for the generic templates

% Extract all the spikes across the recording that are captured by the
% generic templates. Very few real spikes are missed in this way. 
[st3, rez] = standalone_detector(rez, spkTh);


%%
%time, depth, amplitude,[],cluster
% detected depths
% dep = st3(:,2);
% dep = dep - dmin;

Nbatches      = rez.temp.Nbatch;
% which batch each spike is coming from
batch_id = st3(:,5); %ceil(st3(:,1)/dt);

% use_times = [rez.ops.trial_onsets-2.5*ops.fs;rez.ops.trial_onsets+1*ops.fs];
% keep=false(size(st3,1),1);
% for i=1:size(use_times,2)
%     keep(st3(:,1)>use_times(1,i) & st3(:,1) < use_times(2,i))=1;
% end
% st3(~keep,:)=[];
Nspikes_per_batch=arrayfun(@(x)sum(batch_id==x),1:Nbatches);
Nspikes_cutoff=median(Nspikes_per_batch(Nspikes_per_batch>0))/2;
use_batch=Nspikes_per_batch>Nspikes_cutoff;
use_batch(:)=1;

% preallocate matrix of counts with 20 bins, spaced logarithmically
F = zeros(dmax, 20, Nbatches);
for t = 1:Nbatches
    % find spikes in this batch
    ix = find(batch_id==t);
    
    % subtract offset
    dep = st3(ix,2) - dmin;
    
    % amplitude bin relative to the minimum possible value
    amp = log10(min(99, st3(ix,3))) - log10(spkTh);
    
    % normalization by maximum possible value
    amp = amp / (log10(100) - log10(spkTh));
    
    % multiply by 20 to distribute a [0,1] variable into 20 bins
    % sparse is very useful here to do this binning quickly
    M = sparse(ceil(dep/dd), ceil(1e-5 + amp * 20), ones(numel(ix), 1), dmax, 20);    
    
    % the counts themselves are taken on a logarithmic scale (some neurons
    % fire too much!)
     F(:, :, t) = log2(1+M);
%     if length(ix)>Nspikes_cutoff
%         F(:, :, t) = log2(1+M);
%         if init==0
%             F(:, :, 1:t) = repmat(F(:, :, t),1,1,t);
%             init=1;
%         end
%     else
%         if init
%             F(:, :, t) = F(:, :, t-1);
%         end
%     end
end

%%
% determine registration offsets
ysamp = dmin + dd * [1:dmax] - dd/2;
[imin,yblk, F0, F0m] = align_block2(F(:,:,use_batch), ysamp, ops.nblocks);

if isfield(rez, 'F0')
    d0 = align_pairs(rez.F0, F0);
    % concatenate the shifts
    imin = imin - d0;
end

med_filt = getOr(rez.ops, 'drift_estimate_median_filter_pts', 0);
if med_filt>0
    imin_pre_filt=imin;
    imin = medfilt1(imin,med_filt,'truncate');
end
    
%%
if 1%getOr(ops, 'fig', 1)  
%     figure;
%     set(gcf, 'Color', 'w')
%     
%     % plot the shift trace in um
%     
     batch_starts=1:rez.ops.NT:rez.ops.tend;
%     %plot(batch_starts/rez.ops.fs,imin * dd)
%     plot(imin * dd)
%     xlabel('batch number')
%     ylabel('drift (um)')
%     title('Estimated drift traces')
%     drawnow
    
    Maximize(figure(194));clf
    set(gcf, 'Color', 'w')
    % raster plot of all spikes at their original depths
    st_shift = st3(:,2); %+ imin(batch_id)' * dd;
    for j = spkTh:100
        % for each amplitude bin, plot all the spikes of that size in the
        % same shade of gray
        ix = st3(:, 3)==j; % the amplitudes are rounded to integers
        plot(st3(ix, 1)/ops.fs, st_shift(ix), '.', 'color', [1 1 1] * max(0, 1-j/40)) % the marker color here has been carefully tuned
        hold on
    end
    yl=[min(st_shift) max(st_shift)];
    yl=yl+diff(yl)/50.*[-1 1];
    set(gca,'YLim',yl)
    for j=1:length(rez.ops.trial_onsets)
        tsh(j)=plot(rez.ops.trial_onsets([j j])/ops.fs,yl,'--','Color',[.4 .4 .4]);
    end
    run_starts = [1 cumsum(rez.ops.nSamplesBlocks(1:end-1))];
    for i=1:length(run_starts)
        rsh(i)=plot(run_starts([i i])/ops.fs,yl,'-','Color',[.6 .6 .6]);
        str=[rez.ops.runs{i}(8:9),'\_',rez.ops.runs{i}(13:15)];
        th(i)=text(run_starts(i)/ops.fs,yl(2),str,'HorizontalAlignment','left','VerticalAlignment','top');
    end
    uistack(rsh,'bottom')
    uistack(tsh,'bottom')
    
    axis tight

    xlabel('time (sec)')
    ylabel('spike position (um)')
    
    h(1)=gca;
    [h(2),hh]=plot_right(h(1),batch_starts/rez.ops.fs,imin * dd);
    if med_filt>0
        hh2=plot(h(2),batch_starts/rez.ops.fs,imin_pre_filt * dd);
        uistack(hh2,'bottom')
    end
    set(h(2),'YDir','reverse');
    ylabel(h(2),'Estimated drift (um)')
    set(h,'TickDir','out','Box','off')
    set(h(2),'YLim',mean(imin(:) * dd)+diff(yl)/2.*[-1 1])
    title(h(2),'Drift map')
end
%%
% convert to um 
dshift = imin * dd;

% this is not really used any more, should get taken out eventually
[~, rez.iorig] = sort(mean(dshift, 2));

if do_correction
    % sigma for the Gaussian process smoothing
    sig = rez.ops.sig;
    % register the data batch by batch
    dprev = gpuArray.zeros(ops.ntbuff,ops.Nchan, 'single');
    for ibatch = 1:Nbatches
        dprev = shift_batch_on_disk2(rez, ibatch, dshift(ibatch, :), yblk, sig, dprev);
    end
    fprintf('time %2.2f, Shifted up/down %d batches. \n', toc, Nbatches)
else
    fprintf('time %2.2f, Skipped shifting %d batches. \n', toc, Nbatches)
end
% keep track of dshift 
rez.dshift = dshift;
% keep track of original spikes
rez.st0 = st3;

rez.F = F;
rez.F0 = F0;
rez.F0m = F0m;

% next, we can just run a normal spike sorter, like Kilosort1, and forget about the transformation that has happened in here 

%%



