function [WU,similars,iMinChan,xc] = alignWU(WU, ops)

[nt0 , Nchan, Nfilt] = size(WU);
if isfield(ops,'align_similar_clusters') && ops.align_similar_clusters
    
    cutoff_mode='absolute';cutoff=.85;
    cutoff_mode='relative';cutoff=.85;
    cutoff_mode='percent_of_total_clusters';cutoff=15;
    maxlag=nt0*2; %*2 means look for xcs by shifting up to one channel away.
    
    
    %find cross correlations between clusters
    WUrs=reshape(WU, nt0*Nchan, Nfilt);
    xc=zeros(Nfilt);
    
    for i=1:Nfilt
        for j=i+1:Nfilt
            %don't normalize xc so that clusters with big spikes are prioritized
            xc_=xcorr(WUrs(:,i),WUrs(:,j),maxlag);
            xc(i,j)=max(xc_);
        end
    end
    xc=xc./max(xc(:));%norm for better readability
    
    %find similar clusters based on peak cross-correlation
    [xcs,si]=sort(xc(:),'descend');
    status=zeros(size(xcs));
    switch cutoff_mode
        case 'absolute'
            status(xcs>cutoff)=1;
        case 'relative'
            status(xcs>cutoff)=1;
        case 'percent_of_total_clusters'
            status(1:round(Nfilt*cutoff/100))=1;
    end
    
    simi=0;
    similars=cell(0);
    
    while any(status==1)
        testi=find(status==1,1);
        [i1,i2]=ind2sub(size(xc),si(testi));
        matchi=find(cellfun(@(x)any(ismember(x,[i1 i2])),similars));
        if isempty(matchi)
            simi=simi+1;
            similars{simi}=[i1 i2];
            fprintf('Merge %d and %d (xc=%.02f) into new group %d.\n',i1,i2,xcs(testi),simi)
        elseif length(matchi)==1
            fprintf(['Merge %d and %d (xc=%.02f) into group %d (had ',repmat('%d,',1,length(similars{matchi})),').\n'],i1,i2,xcs(testi),matchi,similars{matchi})
            similars{matchi}=unique([similars{matchi} i1 i2]);
        else
            fprintf('%d and %d are similar (xc=%f) but are already in separate similar groups (%d and %d). Skipping.\n',i1,i2,xcs(testi),matchi(1),matchi(2))
        end
        status(testi)=2;
        if any(cellfun(@(y)sum(cellfun(@(x)any(ismember(x,y)),similars))>1,similars))
            error('stop!')
        end
    end
    [mv, imin] = min(reshape(WU, nt0*Nchan, Nfilt), [], 1);
    iMinChan_ = ceil(imin/nt0);
    iMinChan = iMinChan_;
    chan_sign = ones(1,Nfilt);
    similarsO=similars;
    i=0;
    while i < length(similars)
        i=i+1;
        if length(similars{i})==1
            continue
        end
        %go through list of similar clusters, assign each group to have a
        %common channel for alignment (the best channel of the cluaster with
        %the biggest peak)
        [~,biggest_cluster] = min(mv(similars{i}));
        best_chan=iMinChan_(similars{i}(biggest_cluster));
        peak_ratio=zeros(1,length(similars{i}));
        peak_diff=zeros(1,length(similars{i}));
        snr=zeros(1,length(similars{i}));
        similar_channel_peak_sign=zeros(1,length(similars{i}));
        for j=1:length(similars{i})
            peak_ratio(j)=min(WU(:,best_chan,similars{i}(j)))./min(WU(:,iMinChan_(similars{i}(j)),similars{i}(j)));
            [~,min_index_this_best]=min(WU(:,iMinChan_(similars{i}(j)),similars{i}(j)));
            [~,min_index_similars_best]=max(abs(WU(:,best_chan,similars{i}(j))));
            similar_channel_peak_sign(j)=sign(WU(min_index_similars_best,best_chan,similars{i}(j)));
            peak_diff(j)=min_index_this_best-min_index_similars_best;
            snr_inds=min_index_this_best+[-5:5];
            snr_inds(snr_inds<1)=[];
            snr_inds(snr_inds>size(WU,1))=[];
            snr(j)=max(similar_channel_peak_sign(j)*WU(snr_inds,best_chan,similars{i}(j)))./std(WU(:,best_chan,similars{i}(j)));
        end
        %peak_ratio,peak_diff,snr,similar_channel_peak_sign
        keep=abs(peak_diff)<5 & snr > 1;
        if all(~keep)
            keep(1)=true;
        end
        if any(~keep)
            newi=length(similars)+1;
            fprintf(['Removing clusters ',num2str(similars{i}(~keep)),...
                ' from similar group ',num2str(i),' (had ',...
                num2str(similars{i}),', biggest is ',...
                num2str(similars{i}(biggest_cluster)),'),',...
                ' putting in group ',num2str(newi),'\n'])
            similars{newi}=similars{i}(~keep);
            similars{i}(~keep)=[];
        end
        iMinChan(similars{i})=best_chan;
        chan_sign(similars{i})=similar_channel_peak_sign(keep)*-1;
    end
    
    %old ideas
    %     xc2=false(size(xc));xc2(xc>.8)=true;
    %     [mv, imin] = min(reshape(WU, nt0*Nchan, Nfilt), [], 1);
    %     iMinChan_ = ceil(imin/nt0);
    %     iMinChan = zeros(size(iMinChan_));
    %     for i=1:Nfilt
    %         similar_clusters=find(xc2(i,:) | xc2(:,i)');
    %         clusts=[i similar_clusters];
    %         [~,biggest_cluster] = min(mv(clusts));
    %         iMinChan(i)=iMinChan_(clusts(biggest_cluster));
    %     end
    %         WUrs=reshape(WU, nt0*Nchan, Nfilt);
    %     xc3=zeros(Nfilt);
    %     refclust=1:Nfilt;
    
    %     similars=cell(0);
    %     simi=0;
    %     for i=1:Nfilt
    %         for j=i+1:Nfilt
    %             xc_=xcorr(WUrs(:,i),WUrs(:,j),nt0*1,'coeff');
    %             if max(xc_) > 0.8
    %                 matchi=find(cellfun(@(x)any(ismember(x,[i j])),similars));
    %                 if ~isempty(matchi)
    %                     similars{simi}=unique([similars{simi} i j]);
    %                 else
    %                     simi=simi+1;
    %                     similars{simi}=[i j];
    %                 end
    %             end
    %         end
    %     end
    
else
    [~, imin] = min(reshape(WU, nt0*Nchan, Nfilt), [], 1);
    iMinChan = ceil(imin/nt0);
    similars=[];
    xc=[];
end


% imin = rem(imin-1, nt0) + 1;

% [~, imax] = min(W, [], 1);
% dmax = -(imin - 20);
% dmax = min(1, abs(dmax)) .* sign(dmax);

dmax = zeros(Nfilt, 1);
for i = 1:Nfilt
    wu = chan_sign(i)*WU(:,iMinChan(i),i);
    %     [~, imin] = min(diff(wu, 1));
    [~, imin] = min(wu);
    dmax(i) = - (imin- ops.nt0min);
    
    if dmax(i)>0
        WU((dmax(i) + 1):nt0, :,i) = WU(1:nt0-dmax(i),:, i);
    else
        WU(1:nt0+dmax(i),:, i) = WU((1-dmax(i)):nt0,:, i);
    end
end
a=2;
if 0
    i=1;
    figure;ax=subplot1(2,2);
    imagesc(reshape(permute(WU(:,iMinChan(similars{i}(1))+[-4:4],similars{i}),[2 1 3]),9,[]),'Parent',ax(1))
    imagesc(reshape(permute(WUO(:,iMinChan(similars{i}(1))+[-4:4],similars{i}),[2 1 3]),9,[]),'Parent',ax(3))
    imagesc(reshape(permute(WU(:,iMinChan(similars{i}(1))+[-4:4],similars{i}),[1 2 3]),nt0,[])','Parent',ax(2))
    imagesc(reshape(permute(WUO(:,iMinChan(similars{i}(1))+[-4:4],similars{i}),[1 2 3]),nt0,[])','Parent',ax(4))
    
    figure;imagesc(reshape(permute(WUO(:,iMinChan(similars{i}(1))+[-4:4],similars{i}),[2 1 3]),9,[]));title('original')
    figure;imagesc(reshape(permute(WU(:,iMinChan(similars{i}(1))+[-4:4],similars{i}),[2 1 3]),9,[]));title('aligned')
    figure;imagesc(reshape(permute(WUO(:,iMinChan(similars{i}(1))+[-4:4],similars{i}),[1 2 3]),nt0,[])');title('original')
    figure;imagesc(reshape(permute(WU(:,iMinChan(similars{i}(1))+[-4:4],similars{i}),[1 2 3]),nt0,[])');title('aligned')
end
%% extra stuff for algorithm development
%testing if GPU is faster, it's slower!
% if 1
%for i=1:100, tic;xcorr(WUrs(:,1),WUrs(:,2));t(i)=toc; end
%WUrs_GPU=gpuArray(WUrs);
%for i=1:100, tic;xcorr(WUrs_GPU(:,1),WUrs_GPU(:,2));t2(i)=toc; end
% end
if 0
    % for comparing xc methods
    ops={[],'biased','unbiased','coeff'};
    nm=[82 104 8];
    figure;ax=subplot1(4,length(nm));
    for q=1:4
        for r=1:length(nm)
            [xc_,lags]=xcorr(WUrs(:,172),WUrs(:,nm(r)),maxlag,ops{q});
            plot(lags,xc_,'Parent',ax((q-1)*length(nm)+r));
            title(ax((q-1)*length(nm)+r),[num2str(nm(r)),' ',ops{q},' ',num2str(max(xc_))])
        end
        linkaxes(ax((q-1)*length(nm)+[1:length(nm)]))
    end
    set(ax,'Xlim',lags([1 end]))
end

% for plotting similar clusters
%figure;imagesc(reshape(permute(WU(:,:,similars{1}),[2 1 3]),Nchan,[]))

%for comparing two clusters by lineplots
%figure;plot(WU(:,:,82))
%set(gca,'ColorOrderIndex',1)
%hold on;plot(WU(:,:,172),'--')

