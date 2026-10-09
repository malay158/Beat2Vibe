import torch
from torch import nn
import torch.nn.functional as F

class AcousticEncoder(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.cnn=nn.Sequential(
            nn.Conv2d(1,16,5,padding=2),nn.BatchNorm2d(16),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),nn.Flatten())
        self.head=nn.Linear(64,out_dim)
    def forward(self,mel):
        return self.head(self.cnn(mel.unsqueeze(1)))

class VibeLearner(nn.Module):
    def __init__(self):
        super().__init__()
        self.acoustic=AcousticEncoder(128)
        self.project=nn.Sequential(nn.Linear(128,128),nn.ReLU(),nn.Linear(128,64))
    def forward(self,mel):
        return F.normalize(self.project(self.acoustic(mel)),dim=-1)

def contrastive_loss(z1,z2,temp=0.1):
    """Symmetric within-batch InfoNCE. Batch must contain >=2 distinct tracks."""
    if z1.shape[0]<2: raise ValueError('Contrastive learning needs batch size >=2')
    logits=z1@z2.T / temp
    ids=torch.arange(z1.shape[0],device=z1.device)
    return (F.cross_entropy(logits,ids)+F.cross_entropy(logits.T,ids))/2

def augment_spectrogram(mel):
    """Gain perturbation, temporal crop-and-pad, and time-frequency masking (no pitch/tempo shifting)."""
    x=mel.clone()
    b, freq, frames=x.shape
    x=(x+torch.randn(b,1,1,device=x.device)*0.015).clamp(0,1)
    for i in range(b):
        # Temporal cropping and zero padding, without pitch/tempo resampling.
        crop=max(2,frames-frames//10)
        start=int(torch.randint(0,frames-crop+1,(1,),device=x.device))
        segment=x[i,:,start:start+crop].clone()
        x[i].zero_()
        x[i,:,:crop]=segment
        fw=max(1,freq//20); tw=max(1,frames//15)
        f=int(torch.randint(0,freq-fw+1,(1,),device=x.device))
        t=int(torch.randint(0,frames-tw+1,(1,),device=x.device))
        x[i,f:f+fw,:]=0
        x[i,:,t:t+tw]=0
    return x

class Beat2Vibe(nn.Module):
    def __init__(self, tag_dim,meta_dim,n_prototypes=8, variant='full', fixed_weights=(1/3,1/3,1/3)):
        super().__init__()
        self.variant=variant
        self.acoustic=AcousticEncoder(128)
        self.vibe_project=nn.Sequential(nn.Linear(128,128),nn.ReLU(),nn.Linear(128,64))
        self.register_buffer('prototypes',F.normalize(torch.randn(n_prototypes,64),dim=-1))
        self.rhythm=nn.Sequential(nn.Linear(6,64),nn.ReLU(),nn.Linear(64,64))
        self.tags=nn.Sequential(nn.Linear(tag_dim,32),nn.ReLU())
        self.emotions=nn.Sequential(nn.Linear(4,16),nn.ReLU())
        self.metadata=nn.Sequential(nn.Linear(meta_dim,16),nn.ReLU()) if meta_dim else None
        # Disable modalities for controlled ablations without changing the training API.
        self.fusion=nn.Sequential(nn.Linear(128+64+n_prototypes+32+16+16,256),nn.ReLU(),
                                  nn.LayerNorm(256),nn.Dropout(0.2),nn.Linear(256,128))
        self.beat=nn.Sequential(nn.Linear(5,64),nn.ReLU(),nn.Linear(64,32))
        self.classifier=nn.Sequential(nn.Linear(5*128+32+2,256),nn.ReLU(),nn.Dropout(0.2),nn.Linear(256,5))
        self.controller=nn.Sequential(nn.Linear(5*128+32+2,64),nn.ReLU(),nn.Linear(64,3))
        self.register_buffer('rhythm_mu',torch.zeros(5,5))
        self.register_buffer('rhythm_scale',torch.ones(5))
        self.register_buffer('fixed_weights',torch.tensor(fixed_weights,dtype=torch.float32))
    def freeze_ssl(self):
        for m in (self.acoustic,self.vibe_project):
            for p in m.parameters(): p.requires_grad_(False)
            m.eval()
    def train(self,mode=True):
        super().train(mode)
        if mode and not any(p.requires_grad for p in self.acoustic.parameters()):
            self.acoustic.eval();self.vibe_project.eval()
        return self
    def track(self,data,temp_v=0.15):
        mel,rhythm,tags,emotion,metadata=data
        ha=self.acoustic(mel)
        hr=self.rhythm(rhythm)
        z=F.normalize(self.vibe_project(ha),dim=-1)
        vibe=F.softmax(z @ self.prototypes.T / temp_v,dim=-1)
        if self.variant in {'no_vibe','audio_rhythm','audio_only'}: vibe=torch.zeros_like(vibe)
        if self.variant == 'audio_only': hr=torch.zeros_like(hr)
        if self.variant in {'audio_only','audio_rhythm'}:
            tags=torch.zeros_like(tags);emotion=torch.zeros_like(emotion);metadata=torch.zeros_like(metadata)
        tagvec=self.tags(tags)
        emoc=self.emotions(emotion)
        met=self.metadata(metadata) if self.metadata is not None else ha.new_zeros((len(ha),16))
        if self.variant in {'audio_only','audio_rhythm'}:
            tagvec=torch.zeros_like(tagvec);emoc=torch.zeros_like(emoc);met=torch.zeros_like(met)
        inp=torch.cat([ha,hr,vibe,tagvec,emoc,met],dim=1)
        if self.variant in {'audio_only','audio_rhythm'}:
            # Both branches still run through fusion for comparable classifier capacity.
            pass
        return F.normalize(self.fusion(inp),dim=-1)
    def make_delta(self,raw_i,raw_j):
        """Raw rhythm columns: tempo, nbeat, IBI mean/std, onset mean/std."""
        return torch.stack([(raw_j[:,0]-raw_i[:,0])/(raw_i[:,0].clamp(min=1.0)),
            raw_j[:,2]-raw_i[:,2],raw_j[:,3]-raw_i[:,3],
            raw_j[:,4]-raw_i[:,4],raw_j[:,5]-raw_i[:,5]],dim=-1)
    def pair(self,xi,xj,raw_i,raw_j,emo_i,emo_j,desired=None,policy_temp=1.0):
        delta=self.make_delta(raw_i,raw_j)
        # Scaled using only training-pair statistics; same transform for inference.
        d=delta/self.rhythm_scale
        b=self.beat(d)
        if self.variant in {'no_beat','audio_only','audio_rhythm'}: b=torch.zeros_like(b)
        de=(emo_j[:,:2]-emo_i[:,:2])*emo_i[:,2:]*emo_j[:,2:]
        obs=torch.cat([xi,xj,xj-xi,(xj-xi).abs(),xi*xj,b,de],dim=-1)
        logits=self.classifier(obs)
        if self.variant == 'fixed':
            weights=self.fixed_weights.unsqueeze(0).expand(len(obs),-1)
        else:
            weights=F.softmax(self.controller(obs),dim=-1)
        probs=F.softmax(logits,dim=-1)
        sim=F.cosine_similarity(xi,xj)
        distance=(d[:,None,:]-self.rhythm_mu[None,:,:]).abs().sum(dim=-1)
        # All 5 transition-conditioned scores, one per d; select desired when ranking.
        scores=(weights[:,0,None]*probs+weights[:,1,None]*sim[:,None]
                -weights[:,2,None]*distance)
        return {'logits':logits,'weights':weights,'scores':scores,'delta':delta,
                'prob':probs,'obs':obs}

class PairwiseBaseline(nn.Module):
    """Paper Eq. (4-7): separate acoustic/rhythm branches and pairwise MLP."""
    def __init__(self):
        super().__init__()
        self.audio=AcousticEncoder(128)
        self.rhythm=nn.Sequential(nn.Linear(6,64),nn.ReLU(),nn.Linear(64,64))
        self.fusion=nn.Sequential(nn.Linear(192,128),nn.ReLU(),nn.Linear(128,128))
        self.classifier=nn.Sequential(nn.Linear(4*128+4,256),nn.ReLU(),nn.Linear(256,5))
    def track(self,data):
        mel,rhythm,*_=data
        return self.fusion(torch.cat([self.audio(mel),self.rhythm(rhythm)],dim=1))
    def pair(self,xi,xj,r_i,r_j,e_i,e_j):
        va=(e_j[:,:2]-e_i[:,:2])*e_i[:,2:]*e_j[:,2:]
        delta=torch.cat([va,((r_j[:,0]-r_i[:,0])/(r_i[:,0].clamp(min=1.0)))[:,None],
                         (r_j[:,4]-r_i[:,4])[:,None]],dim=1)
        obs=torch.cat([xi,xj,xj-xi,(xj-xi).abs(),delta],dim=1)
        logits=self.classifier(obs)
        return {'logits':logits,'prob':F.softmax(logits,dim=-1)}
