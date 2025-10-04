import os
import warnings
os.environ['OMP_NUM_THREADS'] = '1'
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.cluster._kmeans')

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data = pd.read_csv('dava.csv')
data

print("Veri seti özellikleri:")
print(data.info())
print("\nVeri seti istatistikleri:")
print(data.describe())
print("\nEksik değerler:")
print(data.isnull().sum())

features_for_clustering = ['Case Duration (Days)', 'Number of Witnesses', 'Legal Fees (USD)', 
                          'Number of Evidence Items', 'Severity']

X = data[features_for_clustering].copy()
print(f"\nKümeleme için seçilen özellikler: {features_for_clustering}")
print(f"Seçilen veri boyutu: {X.shape}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertias = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(k_range, inertias, marker='o', linestyle='-', linewidth=2, markersize=8)
plt.title('Elbow Yöntemi - Optimal Küme Sayısı Belirleme')
plt.xlabel('Küme Sayısı (k)')
plt.ylabel('Inertia (WCSS)')
plt.xticks(k_range)
plt.grid(True, alpha=0.3)
plt.show()

print("Elbow yöntemine göre optimal küme sayısını grafikte dirsek noktasından belirleyiniz.")

optimal_k = 3

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)

data_clustered = data.copy()
data_clustered['Cluster'] = cluster_labels

print(f"K-Means kümeleme tamamlandı. {optimal_k} küme oluşturuldu.")
print(f"Her kümedeki veri sayısı:")
print(data_clustered['Cluster'].value_counts().sort_index())

cluster_centers_original = scaler.inverse_transform(kmeans.cluster_centers_)
centers_df = pd.DataFrame(cluster_centers_original, columns=features_for_clustering)
centers_df.index.name = 'Cluster'
print("\nKüme merkezleri (orijinal ölçekte):")
print(centers_df.round(2))

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('K-Means Kümeleme Sonuçları - Özellik Çiftleri', fontsize=16)

feature_pairs = [
    ('Case Duration (Days)', 'Legal Fees (USD)'),
    ('Number of Witnesses', 'Number of Evidence Items'),
    ('Case Duration (Days)', 'Severity'),
    ('Legal Fees (USD)', 'Severity'),
    ('Number of Witnesses', 'Legal Fees (USD)'),
    ('Number of Evidence Items', 'Case Duration (Days)')
]

colors = ['red', 'blue', 'green', 'purple', 'orange']

for i, (x_feature, y_feature) in enumerate(feature_pairs):
    ax = axes[i//3, i%3]
    
    for cluster in range(optimal_k):
        cluster_data = data_clustered[data_clustered['Cluster'] == cluster]
        ax.scatter(cluster_data[x_feature], cluster_data[y_feature], 
                  c=colors[cluster], label=f'Küme {cluster}', alpha=0.7, s=50)
    
    ax.set_xlabel(x_feature)
    ax.set_ylabel(y_feature)
    ax.set_title(f'{x_feature} vs {y_feature}')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n=== KÜME ANALİZİ VE YORUMLARI ===")
for cluster in range(optimal_k):
    cluster_data = data_clustered[data_clustered['Cluster'] == cluster]
    print(f"\n--- KÜME {cluster} ---")
    print(f"Veri sayısı: {len(cluster_data)}")
    print("Ortalama değerler:")
    for feature in features_for_clustering:
        mean_val = cluster_data[feature].mean()
        print(f"  {feature}: {mean_val:.2f}")
    
    outcome_dist = cluster_data['Outcome'].value_counts()
    print(f"Sonuç dağılımı: Aleyhte={outcome_dist.get(0, 0)}, Lehte={outcome_dist.get(1, 0)}")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Kümelere Göre Özellik Dağılımları', fontsize=16)

for i, feature in enumerate(features_for_clustering):
    ax = axes[i//3, i%3]
    
    cluster_data_list = []
    cluster_labels_list = []
    
    for cluster in range(optimal_k):
        cluster_values = data_clustered[data_clustered['Cluster'] == cluster][feature]
        cluster_data_list.append(cluster_values)
        cluster_labels_list.append(f'Küme {cluster}')
    
    ax.boxplot(cluster_data_list, tick_labels=cluster_labels_list)
    ax.set_title(feature)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 8))
plt.imshow(cluster_centers_original.T, cmap='viridis', aspect='auto')
plt.colorbar(label='Değer')
plt.yticks(range(len(features_for_clustering)), features_for_clustering)
plt.xticks(range(optimal_k), [f'Küme {i}' for i in range(optimal_k)])
plt.title('Küme Merkezleri Heatmap')
plt.xlabel('Kümeler')
plt.ylabel('Özellikler')

for i in range(len(features_for_clustering)):
    for j in range(optimal_k):
        plt.text(j, i, f'{cluster_centers_original[j, i]:.1f}', 
                ha='center', va='center', color='white', fontweight='bold')

plt.tight_layout()
plt.show()

print("\n=== GENEL YORUMLAR ===")
print("1. Kümeleme analizi başarıyla tamamlanmıştır.")
print("2. Her kümenin kendine özgü karakteristikleri bulunmaktadır.")
print("3. Kümeleme sonuçları dava türlerini ayırt etmekte faydalı olabilir.")
print("4. Grafikleri inceleyerek her kümenin özelliklerini analiz edebilirsiniz.")