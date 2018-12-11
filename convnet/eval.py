import torch

split_type = 'comp_rxn'

test_dataset = PdosDataset(data_subset = 'test')
model = torch.load('%s_model.pt'%split_type)
model.eval()

y_pred = model(test_dataset.X_data)
MAE = np.abs(y_pred - test_dataset.y_data).mean()

print "Test error for %s split = %.3f"%(split_type,MAE)
