import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('50_Startups.csv')

data.head()

plt.figure(figsize=(10, 6))
plt.scatter(data['R&D Spend'], data['Profit'], alpha=0.7, color='blue')
plt.xlabel('R&D Harcaması ($)')
plt.ylabel('Kâr ($)')
plt.title('R&D Harcaması ve Kâr Arasındaki İlişki')
plt.grid(True, alpha=0.3)
plt.ticklabel_format(style='plain', axis='both')
plt.tight_layout()
plt.show() 

plt.figure(figsize=(10, 6))
plt.scatter(data['Administration'], data['Profit'], alpha=0.7, color='red')
plt.xlabel('Yönetim Harcaması ($)')
plt.ylabel('Kâr ($)')
plt.title('Yönetim Harcamaları ve Kâr Arasındaki İlişki')
plt.grid(True, alpha=0.3)
plt.ticklabel_format(style='plain', axis='both')
plt.tight_layout()
plt.show() 

ortalama_kar = data.groupby('State')['Profit'].mean().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
bars = plt.bar(ortalama_kar.index, ortalama_kar.values, color=['skyblue', 'lightgreen', 'lightcoral'])
plt.xlabel('Eyalet')
plt.ylabel('Ortalama Kâr ($)')
plt.title('Eyaletlere Göre Ortalama Kâr Karşılaştırması')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3, axis='y')
plt.ticklabel_format(style='plain', axis='y')

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'${height:,.0f}',
             ha='center', va='bottom')

plt.tight_layout()
plt.show() 

harcama_verileri = [data['R&D Spend'], data['Administration'], data['Marketing Spend']]
harcama_etiketleri = ['R&D Harcaması', 'Yönetim Harcaması', 'Pazarlama Harcaması']

plt.figure(figsize=(12, 8))
box_plot = plt.boxplot(harcama_verileri, tick_labels=harcama_etiketleri, patch_artist=True)

colors = ['lightblue', 'lightgreen', 'lightcoral']
for patch, color in zip(box_plot['boxes'], colors):
    patch.set_facecolor(color)

plt.ylabel('Harcama Miktarı ($)')
plt.title('Startup Şirketlerinin Harcama Türlerinin Dağılımı')
plt.grid(True, alpha=0.3, axis='y')
plt.ticklabel_format(style='plain', axis='y')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("Harcama Türleri İstatistiksel Özeti:")
print("="*50)
for i, (veri, etiket) in enumerate(zip(harcama_verileri, harcama_etiketleri)):
    print(f"{etiket}:")
    print(f"  Ortalama: ${veri.mean():,.2f}")
    print(f"  Medyan: ${veri.median():,.2f}")
    print(f"  Standart Sapma: ${veri.std():,.2f}")
    print("-" * 30)