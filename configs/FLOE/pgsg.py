# Final PGSG: evidence-conditioned gate + scale-axis communication.
# Same architecture as the historical final model, with single-split data lists.
_base_ = ['./pgsg_evidence_gate.py']

train_options = {
    'segformer_scale_xattn': True,
}