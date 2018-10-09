function ops = convertOpenEphysToRawBInary(ops,do_write)

if(nargin<2)
    do_write=true;
end
%fname       = fullfile(ops.root, sprintf('%s.dat', ops.fbinary));
fname       = ops.fbinary;
UTmkdir(fname);
if(do_write)
    fidout      = fopen(fname, 'w');
    if(fidout==-1)
        error(['Could not open file: ',fname])
    end
end
%

ch=load(ops.chanMap);
chans=sort(ch.chanMap);
fs=cell(ops.Nchan,1);
for j = 1:ops.Nchan
    ops.chanMap_KiloRaw(ch.chanMap==chans(j))=j;
end

for k=1:length(ops.root)
    d=dir(fullfile(ops.root{k}, '*.continuous' ));
    for j = 1:ops.Nchan
        d_ = d(arrayfun(@(x)contains(lower(x.name), sprintf('ch%d.continuous',chans(j))),d));
        [d_.dir]=deal(ops.root{k});
        fs{j} = [fs{j} d_];
    end
end
nblocks = cellfun(@(x) numel(x), fs);
if numel(unique(nblocks))>1
    error('different number of blocks for different channels!')
end
%
nBlocks     = unique(nblocks);
nSamples    = 1024;  % fixed to 1024 for now!

fid = cell(ops.Nchan, 1);
fprintf('Concatenating Open-Ephys data to a single binary file.\n')
do_dc=false;
if isfield(ops,'driftCorrectionFile') && strcmp(ops.driftCorrectionMode,'BeforeFiltering')
    do_dc=true;
    df=load(ops.driftCorrectionFile,'shifts_interp','mean_time');
    fprintf(sprintf('Correcting for drift using %s.\n',ops.driftCorrectionFile))
    if length(df.shifts_interp)~=length(df.mean_time)
        error('fix drift file')
    end
    drift=fastsmooth(df.shifts_interp(4,[1:end,end]),3,2,1);
    drift(1)=0;
    drift_time=[0 df.mean_time(2:end)];
    
    %add end time
    drift_time(end+1)=drift_time(end)*2;%unknown, just make it long
    
    if ops.GPU
        xcoords=round(gpuArray(ops.xc)/10)*10;
        ycoords = gpuArray(ops.yc);
    else
        xcoords=round(ops.xc/10)*10;
        ycoords = ops.yc;
    end
    uxc=unique(xcoords);
    did_2dwarn=zeros(size(uxc));
    [~,reverse_map]=sort(ops.chanMapConn_RecRaw);
    xcoords=xcoords(reverse_map);
    ycoords=ycoords(reverse_map);
    for xi=1:length(uxc)
        ch_inds_=find(xcoords==uxc(xi));
        [~,si]=sort(ycoords(ch_inds_));
        ch_inds{xi}=ch_inds_(si);
    end
    [~,x_order]=sort(cellfun(@length,ch_inds));
    
    if ~isfield(ops,'driftCorrectionInterpMode')
        ops.driftCorrectionInterpMode='1d';
    end
    switch ops.driftCorrectionInterpMode
        case '1d'
            dcim=1;
        case '2d'
            dcim=2;
            xcoords=double(gather(xcoords));
            ycoords=double(gather(ycoords));
        otherwise
            error('unknown driftCorrectionInterpMode')
    end
end
tic
ops.nSamplesBlocks=nan(1,nBlocks);
times=[];
ind=0;
if verLessThan('matlab','8.1')%2013a
    interpfn = @TriScatteredInterp;
else
    interpfn = @scatteredInterpolant;
end
for k = 1:nBlocks
    fprintf(['File ',num2str(k),' of ',num2str(nBlocks),'\n'])
    for j = 1:ops.Nchan
        fid{j}             = fopen(fullfile(fs{j}(k).dir, fs{j}(k).name));
        % discard header information
        fseek(fid{j}, 1024, 0);
    end
    %
    nsamps = 0;
    flag = 1;
    while 1
        ind=ind+1;
        t1=now;
        samples = zeros(nSamples * 1000, ops.Nchan, 'int16');
        for j = 1:ops.Nchan
            collectSamps    = zeros(nSamples * 1000, 1, 'int16');
            
            rawData         = fread(fid{j}, 1000 * (nSamples + 6), '1030*int16', 10, 'b');
            
            %nbatches        = ceil(numel(rawData)/(nSamples+6));
            nbatches        = floor(numel(rawData)/(nSamples+6));
            for s = 1:nbatches
                rawSamps = rawData((s-1) * (nSamples + 6) +6+ [1:nSamples]);
                collectSamps((s-1)*nSamples + [1:nSamples]) = rawSamps;
            end
            samples(:,j)         = collectSamps;
        end
        
        if nbatches<1000
            flag = 0;
        end
        if flag==0
            samples = samples(1:s*nSamples, :);
        end
        
        samples         = samples';
        t2=now;
        v=datevec(t2-t1);
        times(1,ind)=v(4)*3600+v(5)*60+v(6);
        %fprintf('Reading took %3.0fs. \n',v(4)*3600+v(5)*60+v(6));
        if(do_write)
            if(isfield(ops,'common_rejection_mode'))
                t1=now;
                switch ops.common_rejection_mode
                    case 'none'
                    case 'mean'
                        samples=samples-repmat(mean(samples),size(samples,1),1);
                    case 'median'
                        samples=samples-repmat(median(samples),size(samples,1),1);
                    otherwise
                        error(['Unknown comon rejection mode ',ops.common_rejection_mode])
                end
                t2=now;
                v=datevec(t2-t1);
                times(2,ind)=v(4)*3600+v(5)*60+v(6);
                %fprintf('Common Mode Rejection took %3.0fs. \n',v(4)*3600+v(5)*60+v(6));
            end
            if do_dc
                if nsamps==0 && k==1 && 0
                    profile clear;profile on;
                end
                t1=now;
                %dataRAWshifted=nan(size(dataRAW),class(dataRAW));
                drift_interp=interp1(drift_time,drift,nsamps+sum(ops.nSamplesBlocks(1:k-1))+(1:size(samples,2)));
                discretize_factor=10;
                drift_interp=round(drift_interp*discretize_factor)/discretize_factor;
                if any(isnan(drift_interp))
                    error('NaNs!')
                end
                if ops.GPU && dcim==1
                    un_drifts=gpuArray(unique(drift_interp));
                    samples=single(gpuArray(samples));
                elseif dcim==1
                    un_drifts=unique(drift_interp);
                    samples=single(samples);
                elseif dcim==2
                    un_drifts=unique(drift_interp);
                    samples=double(samples);
                end
                
                if dcim==1
                    for xi=x_order
                        %do singleton columns first so that vlaues are
                        %interpolated from data in orginial grid (before
                        %it's been interpolated)
                        if length(ch_inds{xi})==1 && strcmp(ops.chanMap,'/auto/data/code/KiloSort/chanMap_128D_SepColsOffset.mat')
                                if ~did_2dwarn(xi)
                                    warning('There is only one electrode with x position %d. Using 2d interpolation for this one (slow...)',gather(xcoords(ch_inds{xi})))
                                    did_2dwarn(xi)=1;
                                    %dists=sqrt((xcoords-xcoords(ch_inds{xi})).^2 + (ycoords-ycoords(ch_inds{xi})).^2);
                                    %sorted_dists=sort(dists);
                                    %ch_inds2{xi}=find(dists<=sorted_dists(3));
                                    ch_inds2{xi}=find( ( abs(ycoords-ycoords(ch_inds{xi})) < -1*min(drift) + min(diff(sort(ycoords))) ) & abs(xcoords-xcoords(ch_inds{xi}))<=10 );
                                end
                                xc2=gather(xcoords(ch_inds2{xi}));
                                yc2=gather(ycoords(ch_inds2{xi}));
                                xc=gather(xcoords(ch_inds{xi}));
                                yc=gather(ycoords(ch_inds{xi}));
                                samps=double(gather(samples(ch_inds2{xi},:)));
                                if 1
                                    samps2=zeros(1,size(samps,2));
                                    %use linear algebra method for faster results?
                                    parfor ti=1:size(samps,2)
                                        F=interpfn(xc2,yc2,samps(:,ti),'linear','none');%'none' = no extrapolation (outside range -> NaN)
                                        samps2(ti)=F(xc,yc-drift_interp(ti));
                                    end
                                    %load everything onto GPU in one go
                                    samples(ch_inds{xi},:)=samps2;
                                else
                                    for ti=1:size(samps,2)
                                        F=interpfn(xc2,yc2,samps(:,2),'linear','none');%'none' = no extrapolation (outside range -> NaN)
                                        samps(xc2==xc,ti)=F(xc,yc-drift_interp(ti));
                                    end
                                    %load everything onto GPU in one go
                                    samples(ch_inds{xi},:)=samps(xc2==xc,:);
                                end
                        else
                            for ti=1:length(un_drifts)
                                t_inds=drift_interp==un_drifts(ti);
                                samples(ch_inds{xi},t_inds)=interp1(ycoords(ch_inds{xi}),samples(ch_inds{xi},t_inds),ycoords(ch_inds{xi})-un_drifts(ti),'linear',0);
                            end
                        end
                    end
                elseif dcim==2
                    parfor ti=1:size(samples,2)
                        F=interpfn(xcoords,ycoords,samples(:,ti),'linear','none');
                        %F=scatteredInterpolant(xcoords,ycoords,samples(:,ti));
                        samples(:,ti)=F(xcoords,ycoords-drift_interp(ti));
                        
                        %ti=2543;ti=3453;
                        if 0
                            figure;
                            sch=scatter3(xcoords,ycoords,samples(:,ti)+10,100,samples(:,ti),'Filled');
                            sch2=scatter3(xcoords,ycoords,samples(:,ti)+10,100,'k','.');
                            set(sch,'MarkerEdgeColor','k')
                            %ylim([250 550])
                            hold on;
                            xi=min(xcoords):1:max(xcoords);
                            yi=min(ycoords):1:max(ycoords);
                            [X,Y] = meshgrid(xi, yi');
                            F.Method = 'linear';
                            vq1 = F(X,Y);
                            sh=surf(X,Y,vq1);set(sh,'EdgeColor','none')
                        end
                    end
                end
                
                if 0
                    mu = [-30 510]; Sigma = [200 0; 0 200];
                    xi=min(xcoords):20:max(xcoords);
                    yi=min(ycoords):25:max(ycoords);
                    [X1,X2] = meshgrid(xi, yi');
                    s = mvnpdf([X1(:),X2(:)], mu, Sigma);
                    
                    %figure;sh=surf(X1,X2,reshape(s,size(X1)));set(sh,'EdgeColor','none')
                    figure;sh=scatter3(X1(:),X2(:),s,30,s);
                    
                    [X1,X2] = meshgrid(min(xcoords):20:max(xcoords), (min(ycoords):25:max(ycoords))');
                    s2 = mvnpdf([xcoords,ycoords], mu, Sigma);
                    s2=s2./max(s)*10;
                    F=TriScatteredInterp(xcoords,ycoords,s2);
                    si=F(X1(:),X2(:));
                    figure;sh2=scatter3(X1(:),X2(:),si,30,si,'Filled');
                    
                    ys=25;
                    F=TriScatteredInterp(xcoords*2,ycoords,s2);
                    si_shift=F(X1(:),X2(:)-ys);
                    s_shift=F(xcoords*2,ycoords-ys);
                    hold on;sh3=scatter3(X1(:),X2(:)-ys,si_shift,30,si_shift);
                    
                    ms=200;
                    figure;sh2=scatter3(xcoords,ycoords,s2,ms,s2,'Filled');
                    hold on;sh3=scatter3(xcoords,ycoords-ys,s_shift,ms,s_shift,'s','Filled');
                    view([0 90])
                    
                    
                    xtar=-20;ytar=525;
                    inds=[find(ycoords==500 & xcoords==-20) find(ycoords==550 & xcoords==-20) find(ycoords==525 & xcoords==0)];
                    
                    %[b,b_int]=regress(s2(inds),[xcoords(inds) ycoords(inds) ones(length(inds),1)]);
                    %s_int=b(1)*xtar + b(2)*(ytar) + b(3);
                    
                    [b,b_int]=regress(s2(inds),[ones(length(inds),1) xcoords(inds) ycoords(inds) xcoords(inds).*ycoords(inds)]);
                    s_int=b(1)+ b(2)*xtar + b(3)*(ytar) + b(4)*xtar*ytar;
                    
                    F=TriScatteredInterp(xcoords,ycoords,s2);
                    s_shift=F(xtar,ytar);
                    abs(s_int-s_shift)
                    
                    xtar=0;ytar=500;
                    inds=[find(ycoords==500 & xcoords==-20) find(ycoords==475 & xcoords==0) find(ycoords==525 & xcoords==0) find(ycoords==500 & xcoords==20)];
                    [b,b_int]=regress(s2(inds),[ones(length(inds),1) xcoords(inds) ycoords(inds) xcoords(inds).*ycoords(inds)]);
                    s_intxy=b(1)+ b(2)*xtar + b(3)*(ytar) + b(4)*xtar*ytar;
                    [b,b_int]=regress(s2(inds),[ones(length(inds),1) xcoords(inds) ycoords(inds)]);
                    s_int=b(1)+ b(2)*xtar + b(3)*(ytar) ;
                    F=TriScatteredInterp(xcoords,ycoords,s2);
                    s_shift=F(xtar,ytar);
                    abs(s_int-s_shift)
                    
                    sgrid=nan(size(X1));
                    for i=1:length(xcoords)
                        ind1=xcoords(i)==xi;
                        ind2=ycoords(i)==yi;
                        if sum(ind1) > 0 && sum(ind2) >0
                            sgrid(ind2,ind1)=s2(i);
                        end
                    end
                    sgrid_inpaint=inpaint_nans(sgrid);
                end
                
                %save(fn,'shifts','shifts_interp','mean_time','mean_spikes','mean_spikes_interp')
                t2=now;
                v=datevec(t2-t1);
                times(3,ind)=v(4)*3600+v(5)*60+v(6);
                %fprintf('Correcting drift for file %d maxsamp %d took %3.0fs. \n',k,nsamps,v(4)*3600+v(5)*60+v(6));
                
                if nsamps==0 && k==1 && 0
                    profile viewer
                    keyboard
                end
            end
            
            samples=gather_try(samples);
            t1=now;
            fwrite(fidout, samples, 'int16');
            t2=now;
            v=datevec(t2-t1);
            times(4,ind)=v(4)*3600+v(5)*60+v(6);
            %fprintf('Writing took %3.0fs. \n',v(4)*3600+v(5)*60+v(6));
        end
        nsamps = nsamps + size(samples,2);
        
        if flag==0
            break;
        end
    end
    ops.nSamplesBlocks(k) = nsamps;
    
    for j = 1:ops.Nchan
        fclose(fid{j});
    end
    
end
if(do_write)
    fclose(fidout);
end
fprintf('Done concatenating\n')
fn=strrep(ops.fbinary,'.dat','_convertOEPtoRB_times.mat');
save(fn,'times')
disp(num2str(times,'%2.2f  '))