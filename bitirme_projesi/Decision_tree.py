import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("=== VERİ SETİ YÜKLEME ===")
data = pd.read_csv('dava_sonuclari.csv')
print(f"Veri seti boyutu: {data.shape}")
print(f"Sütun adları: {list(data.columns)}")
print("\nİlk 5 satır:")
print(data.head())

print("\n=== VERİ SETİ İNCELEME ===")
print("Veri tipi bilgileri:")
print(data.dtypes)
print(f"\nHer sütundaki boş değer sayısı:")
print(data.isnull().sum())
print(f"\nTemel istatistikler:")
print(data.describe())

print("\n=== VERİ ÖN İŞLEME ===")

print("Kategorik değişken dönüşümü:")
le = LabelEncoder()
data_processed = data.copy()

data_processed['Case Type'] = le.fit_transform(data_processed['Case Type'])
print(f"Case Type kategorileri: {le.classes_}")

print("\nAykırı değer analizi:")
numerical_columns = data_processed.select_dtypes(include=[np.number]).columns.tolist()
for col in numerical_columns:
    Q1 = data_processed[col].quantile(0.25)
    Q3 = data_processed[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data_processed[(data_processed[col] < lower_bound) | (data_processed[col] > upper_bound)]
    print(f"{col}: {len(outliers)} aykırı değer")

print(f"\nHedef değişken (Outcome) dağılımı:")
print(data_processed['Outcome'].value_counts())
print(f"Kazanma oranı: %{(data_processed['Outcome'].sum() / len(data_processed)) * 100:.2f}")

print(f"\nKorelasyon analizi:")
if data_processed['Outcome'].nunique() > 1:
    correlation_matrix = data_processed.corr()
    print(f"Outcome ile en yüksek korelasyona sahip özellikler:")
    outcome_corr = correlation_matrix['Outcome'].abs().sort_values(ascending=False)
    print(outcome_corr.head(6))
else:
    print(f"⚠️  Outcome sütununda sadece tek sınıf var. Korelasyon analizi yapılamıyor.")
    print(f"Mevcut sınıf değeri: {data_processed['Outcome'].unique()[0]}")
    print("Bu durumda makine öğrenmesi modeli anlamlı olmayacaktır.")  

print("\n=== VERİ SETİNİ AYIRMA ===")

X = data_processed.drop('Outcome', axis=1)
y = data_processed['Outcome']

print(f"Özellik sayısı: {X.shape[1]}")
print(f"Toplam veri sayısı: {X.shape[0]}")
print(f"Özellik adları: {list(X.columns)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y  
)

print(f"\nEğitim seti boyutu: {X_train.shape}")
print(f"Test seti boyutu: {X_test.shape}")
print(f"Eğitim setinde kazanma oranı: %{(y_train.sum() / len(y_train)) * 100:.2f}")
print(f"Test setinde kazanma oranı: %{(y_test.sum() / len(y_test)) * 100:.2f}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("\nÖzellik ölçeklendirmesi tamamlandı.")

print("\n=== MODEL KURULUMU ===")

dt_model = DecisionTreeClassifier(
    random_state=42,
    max_depth=10,  
    min_samples_split=5,
    min_samples_leaf=3,
    max_features='sqrt'
)

print("Karar ağacı modeli eğitiliyor...")
dt_model.fit(X_train, y_train)
print("Model eğitimi tamamlandı!")

print("=== MODEL DEĞERLENDİRME ===")

y_train_pred = dt_model.predict(X_train)
y_test_pred = dt_model.predict(X_test)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Eğer sadece bir sınıf varsa precision, recall, f1 hesaplanamaz
if len(np.unique(y_test)) > 1 and len(np.unique(y_test_pred)) > 1:
    train_precision = precision_score(y_train, y_train_pred, zero_division=0)
    test_precision = precision_score(y_test, y_test_pred, zero_division=0)
    
    train_recall = recall_score(y_train, y_train_pred, zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, zero_division=0)
    
    train_f1 = f1_score(y_train, y_train_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
    
    metrics_available = True
else:
    print("⚠️  Tek sınıf olduğu için precision, recall, f1-score hesaplanamıyor.")
    train_precision = test_precision = 0
    train_recall = test_recall = 0  
    train_f1 = test_f1 = 0
    metrics_available = False

print("PERFORMANS METRİKLERİ:")
print("=" * 50)
print(f"{'Metrik':<15} {'Eğitim':<10} {'Test':<10}")
print("-" * 35)
print(f"{'Doğruluk':<15} {train_accuracy:<10.4f} {test_accuracy:<10.4f}")
print(f"{'Precision':<15} {train_precision:<10.4f} {test_precision:<10.4f}")
print(f"{'Recall':<15} {train_recall:<10.4f} {test_recall:<10.4f}")
print(f"{'F1-Score':<15} {train_f1:<10.4f} {test_f1:<10.4f}")

print(f"\nAşırı öğrenme kontrolü:")
if train_accuracy - test_accuracy > 0.1:
    print("⚠️  Model aşırı öğrenme gösteriyor olabilir.")
else:
    print("✅ Model dengeli performans gösteriyor.")

print(f"\nDetaylı Sınıflandırma Raporu (Test Seti):")
if metrics_available and len(np.unique(y_test)) > 1:
    print(classification_report(y_test, y_test_pred, target_names=['Kaybetmek', 'Kazanmak']))
else:
    print("⚠️  Tek sınıf olduğu için detaylı rapor oluşturulamıyor.")
    unique_classes = np.unique(y_test)
    if len(unique_classes) == 1:
        class_name = 'Kaybetmek' if unique_classes[0] == 0 else 'Kazanmak'
        print(f"Tüm örnekler '{class_name}' sınıfına ait.")

print(f"\nKarışıklık Matrisi (Test Seti):")
cm = confusion_matrix(y_test, y_test_pred)
if len(np.unique(y_test)) > 1:
    print(f"Gerçek\\Tahmin  Kaybetmek  Kazanmek")
    print(f"Kaybetmek      {cm[0,0]:<9} {cm[0,1]:<8}")
    print(f"Kazanmak       {cm[1,0]:<9} {cm[1,1]:<8}")
else:
    print(f"⚠️  Tek sınıf olduğu için 2x2 karışıklık matrisi oluşturulamıyor.")
    print(f"Tüm {len(y_test)} örnek doğru tahmin edildi (Kaybetmek sınıfı).")

print(f"\nÖZELLİK ÖNEMLERİ:")
print("=" * 40)
feature_importance = dt_model.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({
    'Özellik': feature_names,
    'Önem': feature_importance
}).sort_values('Önem', ascending=False)

print(importance_df.to_string(index=False, float_format='%.4f'))

print("\n=== SONUÇLARI GÖRSELLEŞTİRME ===")

plt.style.use('default')

plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
top_features = importance_df.head(8)
plt.barh(range(len(top_features)), top_features['Önem'])
plt.yticks(range(len(top_features)), top_features['Özellik'])
plt.xlabel('Önem Skoru')
plt.title('En Önemli 8 Özellik')
plt.gca().invert_yaxis()

plt.subplot(2, 2, 2)
if len(np.unique(y_test)) > 1:
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Karışıklık Matrisi')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Kaybetmek', 'Kazanmak'])
    plt.yticks(tick_marks, ['Kaybetmek', 'Kazanmak'])
    plt.ylabel('Gerçek Etiket')
    plt.xlabel('Tahmin Edilen Etiket')

    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='red', fontweight='bold')
else:
    plt.text(0.5, 0.5, 'Tek Sınıf\nKarışıklık Matrisi\nOluşturulamıyor', 
             ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
    plt.title('Karışıklık Matrisi')
    plt.axis('off')

plt.subplot(2, 2, 3)
if metrics_available:
    metrics = ['Doğruluk', 'Precision', 'Recall', 'F1-Score']
    train_scores = [train_accuracy, train_precision, train_recall, train_f1]
    test_scores = [test_accuracy, test_precision, test_recall, test_f1]

    x = np.arange(len(metrics))
    width = 0.35

    plt.bar(x - width/2, train_scores, width, label='Eğitim', alpha=0.8)
    plt.bar(x + width/2, test_scores, width, label='Test', alpha=0.8)
    plt.xlabel('Metrikler')
    plt.ylabel('Skor')
    plt.title('Model Performans Karşılaştırması')
    plt.xticks(x, metrics, rotation=45)
    plt.legend()
    plt.ylim(0, 1.1)
else:
    plt.text(0.5, 0.5, 'Tek Sınıf\nPerformans Metrikleri\nHesaplanamıyor', 
             ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
    plt.title('Model Performans Karşılaştırması')
    plt.axis('off')

plt.subplot(2, 2, 4)
plot_tree(dt_model, max_depth=3, feature_names=X.columns, 
          class_names=['Kaybetmek', 'Kazanmak'], filled=True, fontsize=8)
plt.title('Karar Ağacı (İlk 3 Seviye)')

plt.tight_layout()
plt.show()

print("\n=== KARAR AĞACI ANALİZİ ===")
print("🌳 KARAR AĞACI NASIL ÇALIŞIR:")
print("-" * 50)
print("1. Karar ağacı, veri setini en iyi ayıran özelliklerle dallanır")
print("2. Her düğümde bir soru sorulur ve veriye göre dallanma yapılır")
print("3. Yapraklar son kararları (Kazanmak/Kaybetmek) temsil eder")

print(f"\n📊 EN ETKİLİ ÖZELLİKLER:")
print("-" * 50)
for i, (_, row) in enumerate(importance_df.head(5).iterrows()):
    print(f"{i+1}. {row['Özellik']}: %{row['Önem']*100:.2f} önem")

print(f"\n🎯 MODEL PERFORMANSI YORUMU:")
print("-" * 50)
print(f"• Test doğruluğu: %{test_accuracy*100:.2f}")

# Tek sınıf problemi kontrolü
if len(np.unique(y)) == 1:
    print("• ⚠️  VERİ SETİ SORUNU: Tüm örnekler aynı sınıfa ait!")
    print("• Bu durumda makine öğrenmesi modeli anlamlı değil.")
    print("• Veri setinde dengeli sınıf dağılımı olmalı.")
else:
    if test_accuracy > 0.85:
        print("• ✅ Çok iyi performans!")
    elif test_accuracy > 0.75:
        print("• ✅ İyi performans!")
    elif test_accuracy > 0.65:
        print("• ⚠️  Orta performans - iyileştirme gerekebilir")
    else:
        print("• ❌ Düşük performans - model revizyonu gerekli")

print(f"\n💡 ÖNERİLER:")
print("-" * 50)
if len(np.unique(y)) == 1:
    print("• VERİ SETİ DÜZELTMESİ GEREKLİ:")
    print("  - Outcome sütununda hem 0 hem 1 değerleri olmalı")
    print("  - Dengeli sınıf dağılımı için veri toplama stratejisi geliştirin")
    print("  - Mevcut veri setinde sadece 'Kaybetmek' sınıfı var")
else:
    print("• En önemli özellikler dava sonucunu belirlemede kritik rol oynuyor")
    print("• Daha fazla veri toplanarak model performansı artırılabilir")
    print("• Hiperparametre optimizasyonu ile model ince ayar edilebilir")
    print("• Özellik mühendisliği ile yeni özellikler türetilebilir")

print(f"\n✅ Analiz tamamlandı! Tüm görevler başarıyla gerçekleştirildi.")
