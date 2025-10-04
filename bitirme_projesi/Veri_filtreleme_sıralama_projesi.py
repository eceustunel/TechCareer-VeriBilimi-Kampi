import pandas as pd

data = pd.read_csv('country.csv')
data

nufusa_gore_azalan = data.sort_values('Population', ascending=False)
print("Nüfusa göre azalan sırada ilk 10 ülke:")
print(nufusa_gore_azalan[['Country', 'Population']].head(10))

gdp_artan = data.sort_values('GDP ($ per capita)', ascending=True)
print("GDP per capita'ya göre artan sırada ilk 10 ülke:")
print(gdp_artan[['Country', 'GDP ($ per capita)']].head(10))

nufus_10m_ustu = data[data['Population'] > 10000000]
print(f"Nüfusu 10 milyonun üzerinde olan ülke sayısı: {len(nufus_10m_ustu)}")
print("\nNüfusu 10 milyonun üzerinde olan ülkeler:")
print(nufus_10m_ustu[['Country', 'Population']].sort_values('Population', ascending=False))

en_yuksek_literacy = data.sort_values('Literacy (%)', ascending=False)
print("En yüksek okur-yazarlık oranına sahip ilk 5 ülke:")
print(en_yuksek_literacy[['Country', 'Literacy (%)']].head(5))

gdp_10k_ustu = data[data['GDP ($ per capita)'] > 10000]
print(f"GDP per capita'sı 10.000'in üzerinde olan ülke sayısı: {len(gdp_10k_ustu)}")
print("\nGDP per capita'sı 10.000'in üzerinde olan ülkeler:")
print(gdp_10k_ustu[['Country', 'GDP ($ per capita)']].sort_values('GDP ($ per capita)', ascending=False))

en_yuksek_nufus_yogunlugu = data.sort_values('Pop. Density (per sq. mi.)', ascending=False)
print("En yüksek nüfus yoğunluğuna sahip ilk 10 ülke:")
print(en_yuksek_nufus_yogunlugu[['Country', 'Pop. Density (per sq. mi.)']].head(10))
