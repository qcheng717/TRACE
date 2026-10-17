"""Detached policy attention monitoring."""
import torch
import torch.nn.functional as F


class PolicyDiagnostics:
    def __init__(self, model):
        self.model = model
        self.metrics = {}
        self.handles = []
        self.hidden = {}
        self.skill_probs = []
        self.neffs = []
        policy = model.policy
        self.handles.append(policy.register_forward_pre_hook(self._condition, with_kwargs=True))
        if policy.proprio_mode == 'current':
            self.handles.append(policy.proprio_input.register_forward_hook(self._current_proprio))
        else:
            self.handles.append(policy.proprio_temporal[-1].register_forward_hook(self._proprio))
        for i, block in enumerate(policy.blocks):
            for j, branch in enumerate(('state', 'visual', 'skill'), 1):
                self.handles.append(block.norms[j].register_forward_pre_hook(
                    lambda m, args, i=i, branch=branch: self.hidden.__setitem__((i, branch), self._action_rms(args[0]))))
                self.handles.append(getattr(block, branch + '_attention').register_forward_hook(
                    lambda m, args, kwargs, out, i=i, branch=branch: self._attention(i, branch, m, args, kwargs, out),
                    with_kwargs=True))

    @staticmethod
    def _rms(x):
        return x.detach().float().square().mean().sqrt()

    def _action_rms(self, x):
        return self._rms(x[self.action_valid])

    def _condition(self, module, args, kwargs):
        condition = args[0] if args else kwargs['condition']
        noisy = args[1] if len(args) > 1 else kwargs['noisy_action']
        mask = args[3] if len(args) > 3 else kwargs.get('action_valid_mask')
        self.action_valid = (torch.ones(noisy.shape[0], noisy.shape[2], dtype=torch.bool, device=noisy.device)
                             if mask is None else mask.detach())
        self.valid = condition.valid.detach()
        self.weights = condition.weights.detach().float()

    @torch.no_grad()
    def _current_proprio(self, module, args, output):
        self.metrics['proprio_cur_rms'] = self._rms(output)

    @torch.no_grad()
    def _proprio(self, module, args, output):
        q = output.detach().float()
        hist = (q * self.valid[..., None]).sum(1) / self.valid.sum(1)[:, None].clamp_min(1)
        cur = q[:, -1]
        self.metrics.update(proprio_hist_rms=self._rms(hist), proprio_cur_rms=self._rms(cur),
                            proprio_hist_cur_cosine=F.cosine_similarity(hist, cur, dim=-1).mean())

    @torch.no_grad()
    def _attention(self, i, branch, module, args, kwargs, output):
        rms = self._action_rms(output[0])
        self.metrics[f'block_{i}.{branch}_attn_out_rms'] = rms
        self.metrics[f'block_{i}.{branch}_attn_to_hidden_ratio'] = rms / (self.hidden[i, branch] + 1e-8)
        if branch == 'state':
            return
        # Reconstruct probabilities in FP32, with actual inputs/weights/mask.
        # Keep production need_weights=False / SDPA unchanged, including BF16.
        with torch.autocast(device_type=args[0].device.type, enabled=False):
            q, k = (x.detach().float() for x in args[:2])
            d = q.shape[-1]; h = module.num_heads; b, t = q.shape[:2]; n = k.shape[1]
            weight = module.in_proj_weight.detach().float()
            bias = module.in_proj_bias.detach().float()
            q = F.linear(q, weight[:d], bias[:d]).reshape(b, t, h, d // h).transpose(1, 2)
            k = F.linear(k, weight[d:2*d], bias[d:2*d]).reshape(b, n, h, d // h).transpose(1, 2)
            raw = q @ k.transpose(-1, -2) / (d // h) ** .5
            logits = raw
            if branch == 'skill':
                mask = kwargs.get('attn_mask')
                prior = torch.zeros_like(raw) if mask is None else mask.detach().float().reshape(b, h, t, n)
                raw_std = raw.transpose(1, 2)[self.action_valid].std(unbiased=False)
                prior_std = prior.masked_fill(~torch.isfinite(prior),0).transpose(1, 2)[self.action_valid].std(unbiased=False)
                for name, value in [('skill_raw_logit_std', raw_std), ('skill_prior_std', prior_std),
                                    ('skill_prior_to_logit_ratio', prior_std / (raw_std + 1e-8))]:
                    self.metrics[f'block_{i}.{name}'] = value
                logits = raw + prior
            prob = logits.softmax(-1)
            if branch == 'skill' and kwargs.get('value_gate') is not None:
                gate = kwargs['value_gate'].detach().float()
                beta = self.model.policy.blocks[i].gate_beta if self.model.policy.blocks[i].router_mode!='local_sparse' else 0.
                for name, val in [('beta', gate.new_tensor(beta)), ('gate_mean',gate.mean()), ('gate_std',gate.std(unbiased=False)), ('gate_min',gate.min()), ('gate_max',gate.max())]:
                    self.metrics[f'block_{i}.{name}'] = val
                v = F.linear(args[2].detach().float(), weight[2*d:], bias[2*d:]).reshape(b,n,h,d//h).transpose(1,2)
                ungated = (prob @ v).transpose(1,2).reshape(b,t,d)
                gated = (prob @ (v*gate[:,None,:,None])).transpose(1,2).reshape(b,t,d)
                self.metrics[f'block_{i}.router_gate_effect'] = self._action_rms(gated-ungated)/(self._action_rms(ungated)+1e-8)
            entropy = -(prob * prob.clamp_min(1e-30).log()).sum(-1)
            self.metrics[f'block_{i}.{branch}_attn_entropy'] = entropy.transpose(1, 2)[self.action_valid].mean()
            self.metrics[f'block_{i}.{branch}_attn_max_prob'] = prob.max(-1).values.transpose(1, 2)[self.action_valid].mean()
            if branch == 'skill':
                self.skill_probs.append(prob.mean(1))
                self.neffs.append(entropy.exp().transpose(1, 2)[self.action_valid])

    def finish(self):
        """Call after backward, before clipping. Removes all hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        metrics = self.metrics
        names = {k.split('.', 1)[1] for k in metrics if k.startswith('block_')}
        for name in sorted(names):
            metrics[name] = torch.stack([v for k, v in metrics.items() if k.startswith('block_') and k.endswith('.' + name)]).mean()
        prob = torch.stack(self.skill_probs).mean(0)  # B,T,K: layer/head average
        mean_prob = prob.masked_fill(~self.action_valid[..., None], 0).sum(1) / self.action_valid.sum(1)[:, None]
        neff = torch.stack(self.neffs)
        adjacent_valid = self.action_valid[:, 1:] & self.action_valid[:, :-1]
        adjacent_tv = (prob[:, 1:]-prob[:, :-1]).abs().sum(-1) / 2
        # A batch of length-one samples has no adjacent action pairs: report zero.
        action_tv = adjacent_tv[adjacent_valid].sum() / adjacent_valid.sum().clamp_min(1)
        metrics.update(skill_attn_neff=neff.mean(), skill_attn_neff_mean=neff.mean(),
                       skill_attn_neff_min=neff.min(), skill_attn_neff_max=neff.max(),
                       router_skill_attn_cosine=F.cosine_similarity(self.weights, mean_prob, dim=-1).mean(),
                       router_skill_attn_tv=(self.weights-mean_prob).abs().sum(-1).mean()/2,
                       skill_attn_action_tv=action_tv)
        groups = {
            'state_cross_attn': lambda n: '.state_attention.' in n,
            'visual_cross_attn': lambda n: '.visual_attention.' in n,
            'skill_cross_attn': lambda n: '.skill_attention.' in n,
            'semantic_adapter': lambda n: n.startswith('semantic_adapter.'),
            'proprio_encoder': lambda n: n.startswith(('policy.proprio_input.', 'policy.proprio_temporal.', 'policy.proprio_position')),
            'skill_basis_projection': lambda n: n.startswith('policy.basis_input.'),
            'router': lambda n: n.startswith(('compatibility.', 'residual.', 'residual_layer', 'semantic_residual.', 'basis_read.', 'basis_projection.', 'score.', 'score_norm.')),
        }
        for group, match in groups.items():
            grads = [p.grad.detach().float().square().sum() for n, p in self.model.named_parameters() if match(n) and p.grad is not None]
            metrics['grad_norm_' + group] = torch.stack(grads).sum().sqrt() if grads else self.weights.new_zeros(())
        return metrics
