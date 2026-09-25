"""
Construction unifiée du modèle CustOMICS.
"""
from src.network.customics import CustOMICS

def build_customics_model(omics_data, sources, params, device,
                           classifier_dim, survival_dim,
                           dropout, num_classes, unsupervised, switch_epoch,
                           linear_decoder=False, lambda_surv=5, domain_params=None,
                           modality_dropout_p=None, md_mode="exclude",
                           optimizer="adam", weight_decay=0.0):
    x_dim = [omics_data[src].shape[1] for src in sources]

    source_params = {
        src: {
            'input_dim': x_dim[i],
            'hidden_dim': params["autoencoder_hidden_dims"][i],
            'latent_dim': params['rep_dim'],
            'norm': True,
            'dropout': params["dropout"],
        }
        for i, src in enumerate(sources)
    }

    central_params = {
        'hidden_dim': params["central_hidden_dims"],
        'latent_dim': params['latent_dim'],
        'norm': True,
        'dropout': params["dropout"],
        'beta': params["beta"],
        'lambda_central': params.get("lambda_central", 1),
    }

    classif_params = {
        'n_class': num_classes,
        'lambda': 0,
        'hidden_layers': classifier_dim,
        'dropout': params["dropout"],
    }

    surv_params = {
        'lambda': lambda_surv,
        'dims': survival_dim,
        'activation': 'SELU',
        'l2_reg': 1e-2,
        'norm': True,
        'dropout': params["dropout"],
    }

    train_params = {'switch': switch_epoch, 'lr': params['lr']}
    if modality_dropout_p is not None:
        train_params['modality_dropout_p'] = modality_dropout_p
        train_params['modality_dropout_mode'] = md_mode

    model = CustOMICS(
        source_params=source_params,
        central_params=central_params,
        classif_params=classif_params,
        surv_params=surv_params,
        train_params=train_params,
        device=device,
        unsupervised=unsupervised,
        linear_central_decoder=linear_decoder,
        domain_params=domain_params,
        optimizer=optimizer,
        weight_decay=weight_decay
    ).to(device)

    return model