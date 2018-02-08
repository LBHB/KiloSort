function [rez, DATA, uproj,ops] = preprocessData(ops,do_write,drift_track_windows_all)

if(nargin<2)
    do_write=true;
end

tic;
% uproj = [];
ops.nt0 	= getOr(ops, {'nt0'}, 61);


if ~isempty(ops.chanMap)
    if ischar(ops.chanMap)
        load(ops.chanMap);
        if isfield(ops,'chanMap2')
            chanMap=ops.chanMap2;
        end
        try
            chanMapConn = chanMap(connected>1e-6);
            xc = xcoords(connected>1e-6);
            yc = ycoords(connected>1e-6);
        catch
            chanMapConn = 1+chanNums(connected>1e-6);
            xc = zeros(numel(chanMapConn), 1);
            yc = [1:1:numel(chanMapConn)]';
        end
        ops.Nchan    = getOr(ops, 'Nchan', sum(connected>1e-6));
        ops.NchanTOT = getOr(ops, 'NchanTOT', numel(connected));
        if exist('fs', 'var')
            ops.fs       = getOr(ops, 'fs', fs);
        end
    else
        if isfield(ops,'chanMap2')
            chanMap=ops.chanMap2;
        else
            chanMap = ops.chanMap;
        end
        chanMapConn = ops.chanMap;
        xc = zeros(numel(chanMapConn), 1);
        yc = [1:1:numel(chanMapConn)]';
        connected = true(numel(chanMap), 1);      
        
        ops.Nchan    = numel(connected);
        ops.NchanTOT = numel(connected);
    end
else
    chanMap  = 1:ops.Nchan;
    connected = true(numel(chanMap), 1);
    
    chanMapConn = 1:ops.Nchan;    
    xc = zeros(numel(chanMapConn), 1);
    yc = [1:1:numel(chanMapConn)]';
end
rez.ops         = ops;
ops.xc=xc;
ops.yc=yc;
ops.chanMapConn=chanMapConn;
switch ops.datatype
    case 'Open-Ephys'
   ops = convertOpenEphysToRawBInary(ops,do_write);  % convert data, only for OpenEphys
    case 'MANTA'
    ops = convertMANTAToRawBinary(ops,do_write);  % convert data, only for MANTA
end


if exist('kcoords', 'var')
    kcoords = kcoords(connected);
else
    kcoords = ones(ops.Nchan, 1);
end
NchanTOT = ops.NchanTOT;
NT       = ops.NT ;


rez.xc = xc;
rez.yc = yc;
rez.xcoords = xcoords;
rez.ycoords = ycoords;
rez.connected   = connected;
rez.ops.chanMap = chanMap;
rez.ops.kcoords = kcoords; 

d = dir(ops.fbinary);
ops.sampsToRead = floor(d.bytes/NchanTOT/2);

if ispc
    dmem         = memory;
    memfree      = dmem.MemAvailableAllArrays/8;
    memallocated = min(ops.ForceMaxRAMforDat, dmem.MemAvailableAllArrays) - memfree;
    memallocated = max(0, memallocated);
else
    memallocated = ops.ForceMaxRAMforDat;
end
nint16s      = memallocated/2;

NTbuff      = NT + 4*ops.ntbuff;
Nbatch      = ceil(d.bytes/2/NchanTOT /(NT-ops.ntbuff));
Nbatch_buff = floor(4/5 * nint16s/rez.ops.Nchan /(NT-ops.ntbuff)); % factor of 4/5 for storing PCs of spikes
Nbatch_buff = min(Nbatch_buff, Nbatch);

%% load data into patches, filter, compute covariance
if isfield(ops,'fslow')&&ops.fslow<ops.fs/2
    [b1, a1] = butter(3, [ops.fshigh/ops.fs,ops.fslow/ops.fs]*2, 'bandpass');
else
    [b1, a1] = butter(3, ops.fshigh/ops.fs*2, 'high');
end

fprintf('Time %3.0fs. Loading raw data... \n', toc);
fid = fopen(ops.fbinary, 'r');
ibatch = 0;
Nchan = rez.ops.Nchan;
if ops.GPU
    CC = gpuArray.zeros( Nchan,  Nchan, 'single');
else
    CC = zeros( Nchan,  Nchan, 'single');
end
if strcmp(ops.whitening, 'noSpikes')
    if ops.GPU
        nPairs = gpuArray.zeros( Nchan,  Nchan, 'single');
    else
        nPairs = zeros( Nchan,  Nchan, 'single');
    end
end
if ~exist('DATA', 'var')
    DATA = zeros(NT, rez.ops.Nchan, Nbatch_buff, 'int16');
end

isproc = zeros(Nbatch, 1);
while 1
    ibatch = ibatch + ops.nSkipCov;
    
    offset = max(0, 2*NchanTOT*((NT - ops.ntbuff) * (ibatch-1) - 2*ops.ntbuff));
    if ibatch==1
        ioffset = 0;
    else
        ioffset = ops.ntbuff;
    end
    fseek(fid, offset, 'bof');
    buff = fread(fid, [NchanTOT NTbuff], '*int16');
    
    %         keyboard;
    
    if isempty(buff)
        break;
    end
    nsampcurr = size(buff,2);
    if nsampcurr<NTbuff
        buff(:, nsampcurr+1:NTbuff) = repmat(buff(:,nsampcurr), 1, NTbuff-nsampcurr);
    end
    if ops.GPU
        dataRAW = gpuArray(buff);
    else
        dataRAW = buff;
    end
    dataRAW = dataRAW';
    dataRAW = single(dataRAW);
    dataRAW = dataRAW(:, chanMapConn);
    
    datr = filter(b1, a1, dataRAW);
    datr = flipud(datr);
    datr = filter(b1, a1, datr);
    datr = flipud(datr);
    
    switch ops.whitening
        case 'noSpikes'
            smin      = my_min(datr, ops.loc_range, [1 2]);
            sd = std(datr, [], 1);
            peaks     = single(datr<smin+1e-3 & bsxfun(@lt, datr, ops.spkTh * sd));
            blankout  = 1+my_min(-peaks, ops.long_range, [1 2]);
            smin      = datr .* blankout;
            CC        = CC + (smin' * smin)/NT;
            nPairs    = nPairs + (blankout'*blankout)/NT;
        otherwise
            CC        = CC + (datr' * datr)/NT;
    end
    
    if ibatch<=Nbatch_buff
        DATA(:,:,ibatch) = gather_try(int16( datr(ioffset + (1:NT),:)));
        isproc(ibatch) = 1;
    end
end
CC = CC / ceil((Nbatch-1)/ops.nSkipCov);
switch ops.whitening
    case 'noSpikes'
        nPairs = nPairs/ibatch;
end
fclose(fid);
fprintf('Time %3.0fs. Channel-whitening filters computed. \n', toc);
switch ops.whitening
    case 'diag'
        CC = diag(diag(CC));
    case 'noSpikes'
        CC = CC ./nPairs;
end

if ops.whiteningRange<Inf
    ops.whiteningRange = min(ops.whiteningRange, Nchan);
    Wrot = whiteningLocal(gather(CC), yc, xc, ops.whiteningRange);
else
    %
    [E, D] 	= svd(CC);
    D = diag(D);
    eps 	= 1e-6;
    Wrot 	= E * diag(1./(D + eps).^.5) * E';
end
Wrot    = ops.scaleproc * Wrot;

fprintf('Time %3.0fs. Loading raw data and applying filters... \n', toc);

fid         = fopen(ops.fbinary, 'r');
fidW    = fopen(ops.fproc, 'w');
if ops.save_filtered_binary
    fidW2    = fopen(strrep(ops.fbinary,'.dat','_filtered.dat'), 'w');
end
if ops.find_drift_correction
    spike_window=round([-.5 1]/1000*ops.fs);
    spike_window=spike_window(1):spike_window(2);
    [all_spikes,spike_time,spike_channel,batch_origin]=UTkilosort_get_spikes(ops,DATA,Nbatch,isproc,spike_window,Wrot);
    sl=strfind(ops.results_path,'/');
    name=ops.results_path(sl(end-1)+1:sl(end)-1);
    fn=sprintf('/auto/users/luke/Projects/MultiChannel/Drift/images/%s/For_drift.mat',name);
    UTmkdir(fn)
    save(fn,'all_spikes','spike_time','spike_channel','batch_origin','ops','Nbatch','Wrot','drift_track_windows_all','-v7.3')
    
    drift_track_windows=[];
    drift_track_windows_start_re_job_onset=[];
    for j=1:length(drift_track_windows_all)
        drift_track_windows=[drift_track_windows ; drift_track_windows_all{j} + sum(ops.nSamplesBlocks(1:j-1))];
        drift_track_windows_start_re_job_onset=[drift_track_windows_start_re_job_onset ; drift_track_windows_all{j} + ops.StartTime_re_Run1(j)*ops.fs];
    end
    
    win_sizes=[.3*20,.3*10,0.3*40,60];
    win_steps=[.3*2,.3*2,0.3*2,1];
    shift_plot_range=[];
    shift_ops.amp_prctile_cutoff=100;
    shift_ops.max_ref_window_range_microns=.5;
    shift_ops.N_ref_windows=30;
    shift_ops.N_ref_history_windows=5;
    shift_ops.N_ref_gap=inf;
    shift_ops.usfac=1000;
    shift_ops.shift_search_bounds_um=[0 0;-150 150];
    shift_ops.suffix_str=' bounded';
    shift_ops.window_snapshots=true;
    %shift_ops.suffix_str=' allspikes';
    for k=1:length(win_sizes)
        [shifts,shifts_interp,mean_time,mean_spikes,mean_spikes_interp,shift_plot_range_]=UTkilosort_calculate_drift_combined(ops,all_spikes,spike_time,spike_channel,drift_track_windows,ops.fs*win_sizes(k),ops.fs*win_steps(k),shift_plot_range,shift_ops);
        if k==1
            shift_plot_range=shift_plot_range_;
        end
        sl=strfind(ops.results_path,'/');
        name=ops.results_path(sl(end-1)+1:sl(end)-1);
        fn=sprintf('/auto/users/luke/Projects/MultiChannel/Drift/images/%s/size %d step %g.mat',name,win_sizes(k),win_steps(k));
        UTmkdir(fn)
        save(fn,'shifts','shifts_interp','mean_time','mean_spikes','mean_spikes_interp')
    end
    if ops.return_after_finding_drift_correction
        return
    end
end

if strcmp(ops.initialize, 'fromData')
    i0  = 0;
    ixt  = round(linspace(1, size(ops.wPCA,1), ops.nt0));
    wPCA = ops.wPCA(ixt, 1:3);
    
    rez.ops.wPCA = wPCA; % write wPCA back into the rez structure
    uproj = zeros(1e6,  size(wPCA,2) * Nchan, 'single');  
end
%
do_dc=false;
if isfield(ops,'driftCorrectionFile') && strcmp(ops.driftCorrectionMode,'AfterFiltering')
    do_dc=true;
    df=load(ops.driftCorrectionFile,'shifts_interp','mean_time');
    if length(df.shifts_interp)~=length(df.mean_time)
        error('fix drift file')
    end
    drift=fastsmooth(df.shifts_interp(4,[1:end,end]),3,2,1);
    drift(1)=0;
    drift_time=[0 df.mean_time(2:end)];     
    drift_time(end+1)=Nbatch*NT;%add end time
    
    xcoords=round(xcoords/10)*10;
    uxc=unique(xcoords);
end
fprintf('\n Batch: 1')
for ibatch = 1:Nbatch
    if mod(ibatch,10)==0
        fprintf(' %d',ibatch)
    end
    if isproc(ibatch) %ibatch<=Nbatch_buff
        if ops.GPU
            datr = single(gpuArray(DATA(:,:,ibatch)));
        else
            datr = single(DATA(:,:,ibatch));
        end
    else
        offset = max(0, 2*NchanTOT*((NT - ops.ntbuff) * (ibatch-1) - 2*ops.ntbuff));
        if ibatch==1
            ioffset = 0;
        else
            ioffset = ops.ntbuff;
        end
        fseek(fid, offset, 'bof');
        
        buff = fread(fid, [NchanTOT NTbuff], '*int16');
        if isempty(buff)
            break;
        end
        nsampcurr = size(buff,2);
        if nsampcurr<NTbuff
            buff(:, nsampcurr+1:NTbuff) = repmat(buff(:,nsampcurr), 1, NTbuff-nsampcurr);
        end
        
        if ops.GPU
            dataRAW = gpuArray(buff);
        else
            dataRAW = buff;
        end
        dataRAW = dataRAW';
        dataRAW = single(dataRAW);
        dataRAW = dataRAW(:, chanMapConn);
        
        datr = filter(b1, a1, dataRAW);
        datr = flipud(datr);
        datr = filter(b1, a1, datr);
        datr = flipud(datr);
        
        datr = datr(ioffset + (1:NT),:);
    end
    
    datr    = datr * Wrot;
    if ops.GPU
        dataRAW = gpuArray(datr);
    else
        dataRAW = datr;
    end
    if do_dc
        
        if ibatch==1
            profile clear;profile on;
        end
        t1=now;
        dataRAW=dataRAW';
        %dataRAWshifted=nan(size(dataRAW),class(dataRAW));
        drift_interp=interp1(drift_time,drift,(ibatch-1)*NT+(1:NT));
         discretize_factor=10;
        drift_interp=round(drift_interp*discretize_factor)/discretize_factor;        
         if any(isnan(drift_interp))
             error('NaNs!')
         end
        if ops.GPU
            y = gpuArray(rez.ycoords);
            un_drifts=gpuArray(unique(drift_interp));
        else
            y = rez.ycoords;
            un_drifts=unique(drift_interp);
        end
        for xi=1:length(uxc)
            ch_inds=xcoords==uxc(xi);
            if 1
                for ti=1:length(un_drifts)
                    t_inds=drift_interp==un_drifts(ti);
                    dataRAW(ch_inds,t_inds)=interp1(y(ch_inds),dataRAW(ch_inds,t_inds),y(ch_inds)-un_drifts(ti),'linear',0);
                end
            else
                for ti=1:NT
                    %df.mean_time;
                    %dataRAWshifted(ti,ch_inds)=interp1(rez.ycoords(ch_inds),dataRAW(ti,ch_inds),rez.ycoords(ch_inds)-drift_interp(ti));
                    dataRAW(ch_inds,ti)=interp1(rez.ycoords(ch_inds),dataRAW(ch_inds,ti),rez.ycoords(ch_inds)+drift_interp(ti),'linear',0);
                end
            end
        end
        %save(fn,'shifts','shifts_interp','mean_time','mean_spikes','mean_spikes_interp')
        dataRAW=dataRAW';
        t2=now;
        v=datevec(t2-t1);
        fprintf('Correcting drift for batch %d took %3.5gs. \n',ibatch,v(4)*3600+v(5)*60+v(6));
        
        if ibatch==1
            profile viewer
            keyboard
        end
    end
    if ops.save_filtered_binary
        if ibatch==1
            fwrite(fidW2, gather_try(int16(datr(1:(NT-ops.ntbuff),:)')), 'int16');
        else
            %fwrite(fidW2, datcpu(ops.ntbuff:NT,:)', 'int16');
            fwrite(fidW2, gather_try(int16(datr((ops.ntbuff+1):(NT),:)')), 'int16');
        end
    end  
    if do_dc
        if ibatch<=Nbatch_buff
            DATA(:,:,ibatch) = gather_try(dataRAW);
        else
            datcpu  = gather_try(int16(dataRAW));
            fwrite(fidW, datcpu, 'int16');
        end
    else
        if ibatch<=Nbatch_buff
            DATA(:,:,ibatch) = gather_try(datr);
        else
            datcpu  = gather_try(int16(datr));
            fwrite(fidW, datcpu, 'int16');
        end
    end

    dataRAW = single(dataRAW);
    dataRAW = dataRAW / ops.scaleproc;
    if strcmp(ops.initialize, 'fromData') %&& rem(ibatch, 10)==1
        % find isolated spikes
        [row, col, mu] = isolated_peaks(dataRAW, ops.loc_range, ops.long_range, ops.spkTh);
        
        % find their PC projections
        uS = get_PCproj(dataRAW, row, col, wPCA, ops.maskMaxChannels);
        
        uS = permute(uS, [2 1 3]);
        uS = reshape(uS,numel(row), Nchan * size(wPCA,2));
        
        if i0+numel(row)>size(uproj,1)
            uproj(1e6 + size(uproj,1), 1) = 0;
        end
        
        uproj(i0 + (1:numel(row)), :) = gather_try(uS);
        i0 = i0 + numel(row);
    end
    

end
if strcmp(ops.initialize, 'fromData')
   uproj(i0+1:end, :) = []; 
end
Wrot        = gather_try(Wrot);
rez.Wrot    = Wrot;

fclose(fidW);
if ops.save_filtered_binary
    fclose(fidW2);
end
fclose(fid);
if ops.verbose
    fprintf('Time %3.2f. Whitened data written to disk... \n', toc);
    fprintf('Time %3.2f. Preprocessing complete!\n', toc);
end


rez.temp.Nbatch = Nbatch;
rez.temp.Nbatch_buff = Nbatch_buff;

if 0
    %bin by spike index
    Nsp=10000;
    sti=1:Nsp:(size(uproj,1)-Nsp);
    mean_clips=zeros(size(wPCA,1), Nchan,length(sti),'single');
    for i=1:length(sti)
        %to get waveforms
        Us = reshape(uproj(sti(i)+[0:Nsp-1],:),Nsp, Nchan,size(wPCA,2));
        clips_back=reshape(wPCA*reshape(permute(Us,[3 1 2]),3,[]),size(wPCA,1), size(Us,1), Nchan);
        mean_clips(:,:,i)=mean(clips_back,2);
    end
    
    figure;
    for i=1:length(sti)
        imagesc(mean_clips(:,:,i)')
        title(num2str(i))
        pause
    end
    
    %bin by batch
    mean_clips2=zeros(size(wPCA,1), Nchan,Nbatch,'single');
    for i=1:Nbatch
        %to get waveforms
        Us = reshape(uproj(batch_origin==i,:),sum(batch_origin==i), Nchan,size(wPCA,2));
        clips_back=reshape(wPCA*reshape(permute(Us,[3 1 2]),3,[]),size(wPCA,1), size(Us,1), Nchan);
        mean_clips2(:,:,i)=mean(clips_back,2);
    end
    
    figure;
    for i=1:Nbatch
        imagesc(mean_clips3(:,:,i)')
        title(num2str(i))
        pause
    end
    
    %bin by trial (silent periods only)
    windows_per_drift_window=3;
    N=floor(size(drift_track_windows,1)/windows_per_drift_window);
    mean_clips3=zeros(size(wPCA,1), Nchan,N,'single');
    for i=1:N
        %to get waveforms
        inds=false(size(spike_time));
        for j=1:windows_per_drift_window
            inds=inds | spike_time>=drift_track_windows(i+j-1,1) & spike_time<drift_track_windows(i+j-1,2);
        end
        Us = reshape(uproj(inds,:),sum(inds), Nchan,size(wPCA,2));
        clips_back=reshape(wPCA*reshape(permute(Us,[3 1 2]),3,[]),size(wPCA,1), size(Us,1), Nchan);
        mean_clips3(:,:,i)=mean(clips_back,2);
        k=32;
        [clusts,c] = kmeans(uproj(inds,:),k,'EmptyAction','singleton');
        Us = reshape(c,k, Nchan,size(wPCA,2));
        clips_back=reshape(wPCA*reshape(permute(Us,[3 1 2]),3,[]),size(wPCA,1), size(Us,1), Nchan);
        mean_clips4(:,:,:,i)=permute(clips_back,[1 3 2]);
        subz=UTget_subs(k);
        figure;ax=subplot1(subz{:});
        clear dat
        xl=[.5 diff(drift_track_windows(1,:))+extra+.5];yl=[.5 size(mean_clips3,2)+.5];
        for i=1:length(ind)
            bi=floor(drift_track_windows(i,:)/NT)+1;
            ti=mod(drift_track_windows(i,:),NT);
            if bi(1)==bi(2)
                dat(:,:,i)=DATA(ti(1):ti(2)+extra,:,bi(1));
            else
                dat(:,:,i)=[DATA(ti(1):end,:,bi(1));DATA(1:ti(2)+extra,:,bi(2))];
            end
            imagesc(dat(:,:,i)','Parent',ax(i));
            set(ax(i),'YDir','normal','XLim',xl,'YLim',yl)
            text(xl(1),yl(2),num2str(ind(i)),'VerticalAlignment','top','Parent',ax(i),'Color','w')
        end
        cl=[min(dat(:)) max(dat(:))];
        set(ax,'CLim',cl)
    end
    
    
    ind=1:12;
    subz=UTget_subs(length(ind));
    figure;ax=subplot1(subz{:});
    xl=[.5 size(mean_clips3,1)+.5];yl=[.5 size(mean_clips3,2)+.5];
    for i=1:length(ind)
        imagesc(mean_clips3(:,:,ind(i))','Parent',ax(i));
        set(ax(i),'YDir','normal','XLim',xl,'YLim',yl)
        text(xl(1),yl(2),num2str(ind(i)),'VerticalAlignment','top','Parent',ax(i),'Color','w')
    end
    cl=[min(min(min(mean_clips3(:,:,ind)))) max(max(max(mean_clips3(:,:,ind))))];
    set(ax,'CLim',cl)
    
    ind=1:12;
    subz=UTget_subs(length(ind));
    figure;ax=subplot1(subz{:});
    clear dat
    extra=30000*.1;
    xl=[.5 diff(drift_track_windows(1,:))+extra+.5];yl=[.5 size(mean_clips3,2)+.5];
    for i=1:length(ind)
        bi=floor(drift_track_windows(i,:)/NT)+1;
        ti=mod(drift_track_windows(i,:),NT);
        if bi(1)==bi(2)
            dat(:,:,i)=DATA(ti(1):ti(2)+extra,:,bi(1));
        else
            dat(:,:,i)=[DATA(ti(1):end,:,bi(1));DATA(1:ti(2)+extra,:,bi(2))];
        end
        imagesc(dat(:,:,i)','Parent',ax(i));
        set(ax(i),'YDir','normal','XLim',xl,'YLim',yl)
        text(xl(1),yl(2),num2str(ind(i)),'VerticalAlignment','top','Parent',ax(i),'Color','w')
    end
    cl=[min(dat(:)) max(dat(:))];
    set(ax,'CLim',cl)
    
    ind=1:12;
    subz=UTget_subs(length(ind));
    figure;ax=subplot1(subz{:});
    clear dat
    extra=30000*.1;
    xl=[.5 diff(drift_track_windows(1,:))+extra+.5];yl=[.5 size(mean_clips3,2)+.5];
    ch=[29 30 31 32];
    for i=1:length(ind)
        bi=floor(drift_track_windows(i,:)/NT)+1;
        ti=mod(drift_track_windows(i,:),NT);
        if bi(1)==bi(2)
            dat(:,:,i)=DATA(ti(1):ti(2)+extra,:,bi(1));
        else
            dat(:,:,i)=[DATA(ti(1):end,:,bi(1));DATA(1:ti(2)+extra,:,bi(2))];
        end
        plot(dat(:,ch,i),'Parent',ax(i));
    end
    set(ax,'XLim',[1 size(dat,1)])
    set(ax,'YLim',[min(arrayfun(@(x)min(get(x,'YLim')),ax)) max(arrayfun(@(x)max(get(x,'YLim')),ax))])
    
    ind=1:12;
    subz=UTget_subs(length(ind));
    figure;ax=subplot1(subz{:});
    clear dat
    extra=30000*.1;
    xl=[.5 diff(drift_track_windows(1,:))+extra+.5];yl=[.5 size(mean_clips3,2)+.5];
    for i=1:length(ind)
        inds=spike_time>=drift_track_windows(ind(i),1) & spike_time<drift_track_windows(ind(i),2);
        Us = reshape(uproj(inds,:),sum(inds), Nchan,size(wPCA,2));
        clips_back=reshape(wPCA*reshape(permute(Us,[3 1 2]),3,[]),size(wPCA,1), size(Us,1), Nchan);
        [mv,ch]=min(min(clips_back,[],1),[],3);
        hist(ax(i),ch,1:64);
        c(i)=length(ch);
    end
    set(ax,'YLim',[0 max(arrayfun(@(x)max(get(x,'YLim')),ax))])
    set(ax,'XLim',[0 65])
    
    
    
    figure;
    for i=1:size(mean_clips3,3)
        imagesc(mean_clips3(:,:,i)')
        title(num2str(i))
        pause
    end
    
    Us = reshape(uproj,size(uproj,1), Nchan,size(wPCA,2));
    figure;plot(squeeze(Us(:,32,:)))
    
    
    Us = permute(uS,[2 1 3]);
    clips_back=reshape(wPCA*reshape(permute(Us,[3 1 2]),3,[]),size(wPCA,1), size(Us,1), Nchan);
    
    [nT, nChan] = size(dataRAW);
    dt = -21 + [1:size(wPCA,1)];
    inds = repmat(row', numel(dt), 1) + repmat(dt', 1, numel(row));
    
    clips = reshape(dataRAW(inds, :), numel(dt), numel(row), nChan);
end
end