"""EMA of trainable model parameters; frozen backbone is read from raw state."""
import torch

class ModelEMA:
    def __init__(self, model, decay):
        if not 0<decay<1:raise ValueError('EMA decay must be between zero and one')
        self.decay=decay;self.updates=0
        self.shadow={k:p.detach().clone() for k,p in model.named_parameters() if p.requires_grad}
    @torch.no_grad()
    def update(self,model):
        for k,p in model.named_parameters():
            if k in self.shadow:self.shadow[k].lerp_(p.detach(),1-self.decay)
        self.updates+=1
    def state_dict(self):return dict(decay=self.decay,updates=self.updates,shadow=self.shadow)
    def load_state_dict(self,state):
        if state['decay']!=self.decay or set(state['shadow'])!=set(self.shadow):raise ValueError('EMA configuration mismatch')
        for k,v in self.shadow.items():v.copy_(state['shadow'][k])
        self.updates=state['updates']

def inference_state(payload,use_ema=True):
    state=dict(payload['model'])
    if use_ema and 'ema' in payload:state.update(payload['ema']['shadow'])
    return state
