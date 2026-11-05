# Single-split PGSG evidence + gate ablation; no DANN or PBA.
# Uses train_417.json / val_417.json. Historical paper fold-0 metrics do not apply.
_base_ = ['./sota_segformer_in1k_dlr_417.py']

train_options = {
    'segformer_scale_gate': True,
    'segformer_gate_mode': 'softmax',
    'segformer_gate_evidence': True,
    'segformer_scale_xattn': False,
    'segformer_dann': False,
    'segformer_pba': False,
}