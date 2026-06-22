import numpy as np
from sklearn.datasets import make_regression
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import json

SEMENTE = 3737

torch.manual_seed(SEMENTE)
np.random.seed(SEMENTE)

N_FEAT_INFO = [5,10,20,30,40,50,60,70,80,90,100]

RMSE_RF_PCA =[]
RMSE_RF_AE = []

RMSE_SVR_PCA = []
RMSE_SVR_AE = []

RMSE_LR_PCA = []
RMSE_LR_AE = []

########################### Definição do Autoencoder ###########################

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(100, 75),
            nn.ReLU(), 
            nn.Linear(75, 50), 
            nn.ReLU(), 
            nn.Linear(50, 25), 
            nn.ReLU(), 
            nn.Linear(25, 10), 
        )

        self.decoder = nn.Sequential(
            nn.Linear(10, 25),
            nn.ReLU(), 
            nn.Linear(25, 50), 
            nn.ReLU(), 
            nn.Linear(50, 75), 
            nn.ReLU(), 
            nn.Linear(75, 100), 
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

model = Autoencoder()
criterion = nn.MSELoss()
optimizer  = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay = 1e-5)

########################### Loop com os testes ###########################

for n_info in N_FEAT_INFO:
    
    X, y = make_regression(
        n_samples=5000,
        n_features=100,
        n_informative=n_info,
        noise=5,
        random_state=SEMENTE
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=SEMENTE
    )

    pca = PCA(n_components=10)

    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor)
    test_dataset = TensorDataset(X_test_tensor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False
    )

    num_epochs = 100

    for epoch in range(num_epochs):

        for (features, ) in train_loader:

            enc, dec = model(features)

            loss = criterion(dec, features)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # print(f'Epoch:{epoch+1}, Loss:{loss.item():.4f}')

    with torch.no_grad():

        X_train_encoded = model.encoder(X_train_tensor)

        X_test_encoded = model.encoder(X_test_tensor)

    X_train_encoded = X_train_encoded.numpy()
    X_test_encoded = X_test_encoded.numpy()

    print(f"Treinando autoencoder com {n_info} features informativas")
    
    ########################### Floresta Aleatória ###########################

    model_RF = RandomForestRegressor()

    model_RF.fit(X_train_pca, y_train)

    y_pred_PCA = model_RF.predict(X_test_pca)

    RMSE = root_mean_squared_error(y_test, y_pred_PCA)

    RMSE_RF_PCA.append(RMSE)

    model_RF.fit(X_train_encoded, y_train)

    y_pred_AE = model_RF.predict(X_test_encoded)

    RMSE = root_mean_squared_error(y_test, y_pred_AE)

    RMSE_RF_AE.append(RMSE)

    ########################### Support Vector Regressor ###########################

    model_svr = SVR(
    kernel='rbf'
    )

    model_svr.fit(X_train_pca, y_train)

    y_pred_svr = model_svr.predict(X_test_pca)

    RMSE = root_mean_squared_error(y_test, y_pred_svr)

    RMSE_SVR_PCA.append(RMSE)

    model_svr.fit(X_train_encoded, y_train)

    y_pred_svr = model_svr.predict(X_test_encoded)

    RMSE = root_mean_squared_error(y_test, y_pred_svr)

    RMSE_SVR_AE.append(RMSE)

    ########################### Regressão Linear ###########################

    model_LR = LinearRegression()

    model_LR.fit(X_train_pca, y_train)

    y_pred_LR = model_LR.predict(X_test_pca)

    RMSE = root_mean_squared_error(y_test, y_pred_LR)

    RMSE_LR_PCA.append(RMSE)

    model_LR.fit(X_train_encoded, y_train)

    y_pred_LR = model_LR.predict(X_test_encoded)

    RMSE = root_mean_squared_error(y_test, y_pred_LR)

    RMSE_LR_AE.append(RMSE)

    ########################### Plotar gráfico ###########################

plt.figure(figsize=(10,6))

plt.plot(N_FEAT_INFO, RMSE_RF_PCA, marker = 'o', label='RF + PCA')
plt.plot(N_FEAT_INFO, RMSE_RF_AE, marker = 's', label='RF + AE')

plt.plot(N_FEAT_INFO, RMSE_SVR_PCA, marker = 'o', label='SVR + PCA')
plt.plot(N_FEAT_INFO, RMSE_SVR_AE, marker = 's', label='SVR + AE')

plt.plot(N_FEAT_INFO, RMSE_LR_PCA, marker = 'o', label='LR + PCA')
plt.plot(N_FEAT_INFO, RMSE_LR_AE, marker = 's', label='LR + AE')

plt.xlabel('Número de Features Informativas')
plt.ylabel('RMSE')

plt.title('Comparação PCA vs Autoencoder com ruído')

plt.legend()

plt.grid(True)

plt.savefig('comparacao_pca_autoencoder_ruido_5.png', dpi=300)

plt.show()

########################### Salvar resultados ###########################

dados = {
    "RF_PCA": RMSE_RF_PCA,
    "RF_AE": RMSE_RF_AE,
    "SVR_PCA": RMSE_SVR_PCA,
    "SVR_AE": RMSE_SVR_AE,
    "LR_PCA": RMSE_LR_PCA,
    "LR_AE": RMSE_LR_AE   
}

with open("resultados_ruido_5.json", "w") as f:
    json.dump(dados, f)