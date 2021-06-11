##############################################################
# BG-NBD ve Gamma-Gamma ile CLTV Prediction
from sqlalchemy import create_engine
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
import mysql.connector
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)


def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


#########################
# Verinin Veri Tabanından Okunması
#########################

creds = {'user': '',
         'passwd': '',
         'host': '',
         'port': '',
         'db': ''
         }

# MySQL conection string.
connstr = 'mysql+mysqlconnector://{user}:{passwd}@{host}:{port}/{db}'
conn = create_engine(connstr.format(**creds))
# conn.close()
retail_mysql_df = pd.read_sql_query("select * from online_retail_2010_2011", conn)

type(retail_mysql_df)

retail_mysql_df.shape
retail_mysql_df.head()
retail_mysql_df.info()
df = retail_mysql_df.copy()
#########################
# Veri Ön İşleme
#########################
df.describe().T
df = df[df["Country"] == "United Kingdom"]
df.dropna(inplace=True)
df = df[~df["Invoice"].str.contains("C", na=False)]
df = df[df["Quantity"] > 0]


replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")

# Ön İşleme Sonrası
df.describe().T

df["TotalPrice"] = df["Quantity"] * df["Price"]


#########################
# Lifetime Veri Yapısının Hazırlanması
########################


today_date = dt.datetime(2011, 12, 11)
cltv_df = df.groupby('Customer ID').agg({'InvoiceDate': [lambda date: (date.max() - date.min()).days,
                                                         lambda date: (today_date - date.min()).days],
                                         'Invoice': lambda num: num.nunique(),
                                         'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

cltv_df.columns = cltv_df.columns.droplevel(0)
cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']
cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]
cltv_df = cltv_df[cltv_df["monetary"] > 0]
cltv_df.head()
cltv_df["recency"] = cltv_df["recency"] / 7
cltv_df["T"] = cltv_df["T"] / 7
cltv_df = cltv_df[(cltv_df['frequency'] > 1)]



##############################################################
# Modelinin Kurulması
##############################################################

def model_bgf_ggf(cltv_df):
    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T'])
    cltv_df["expected_purc_1_week"] = bgf.predict(1,
                                                  cltv_df['frequency'],
                                                  cltv_df['recency'],
                                                  cltv_df['T'])
    cltv_df["expected_purc_1_month"] = bgf.predict(4,
                                                   cltv_df['frequency'],
                                                   cltv_df['recency'],
                                                   cltv_df['T'])
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(cltv_df['frequency'], cltv_df['monetary'])
    cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df['frequency'], cltv_df['monetary'])

    return ggf, bgf

ggf, bgf = model_bgf_ggf(cltv_df)

def cltv_customer_lifetime_value(time):
    cltv = ggf.customer_lifetime_value(bgf,
                                       cltv_df['frequency'],
                                       cltv_df['recency'],
                                       cltv_df['T'],
                                       cltv_df['monetary'],
                                       time=time,  # 3 aylık
                                       freq="W",  # T'nin frekans bilgisi.
                                       discount_rate=0.01)

    cltv = cltv.reset_index()
    cltv.sort_values(by="clv", ascending=False).head(50)
    cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")

    cltv_final.sort_values(by="clv", ascending=False).head(10)

    # CLTV'nin Standartlaştırılması
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(cltv_final[["clv"]])
    cltv_final["scaled_clv"] = scaler.transform(cltv_final[["clv"]])

    # Sıralayalım:
    cltv_final.sort_values(by="scaled_clv", ascending=False).head()

    return cltv_final

###########################################
# Görev 1: 6 aylık CLTV Prediction

cltv_6M = cltv_customer_lifetime_value(6)

cltv_6M.head()
cltv_6M.corr()

# Uzun vadede frekansı yüksek olan müşterinin clv değerinin daha yüksek olduğu,
# bıraktığı monetary değeriden ziyade sürekliliğin sağlamasının daha önemli olduğunu gözlemlenmektedir.

###########################################
# Görev 2: Farklı zaman periyotlarından oluşan CLTV analizi
# 2010-2011 UK müşterileri için 1 aylık ve 12 aylık CLTV hesaplayınız.
# 1 aylık CLTV'de en yüksek olan 10 kişi ile 12 aylık'taki en yüksek 10 kişiyi analiz ediniz.
# Fark var mı? Varsa sizce neden olabilir?


cltv_1M = cltv_customer_lifetime_value(1)
cltv_1M.head()
cltv_12M = cltv_customer_lifetime_value(12)
cltv_12M.head()

cltv_1M.sort_values(by='clv', ascending=False).head(10)
cltv_12M.sort_values(by='clv', ascending=False).head(10)

# Müşteri sıralamalarında değişiklik olmuştur.
# 587 index 14088.0000 nolu müşteri numarasına sahip kişiye baktığımızda
# Kısa vade için expected_average_profit değeri etkili olsada 12 aylık tahminde
# frekans değeri daha yüksek olan 406 index 13694.0000 nolu müşteri ile yer değiştirmiştir.
# Buradan da gözlemlediğimiz üzere uzun vadede kazanç önemini yitirmektedir.



###########################################
# Görev 3: 2010-2011 UK müşterileri için 6 aylık CLTV'ye göre tüm müşterilerinizi 4 gruba
# (segmente) ayırınız ve grup isimlerini veri setine ekleyiniz.
# 4 grup içerisinden seçeceğiniz 2 grup için yönetime kısa kısa 6 aylık aksiyon
# önerilerinde bulununuz


def cltv_segmentaion(cltv_final):
    cltv_final["segment"] = pd.qcut(cltv_final["scaled_clv"], 4, labels=["D", "C", "B", "A"])
    return cltv_final

cltv_6M = cltv_customer_lifetime_value(6)
cltv_6M_segment = cltv_segmentaion(cltv_6M)

cltv_6M_segment.iloc[:, 1:].groupby("segment").agg(
    {"sum", "count", "mean", "median"})

# C ve D segmentlerinin T,frequency, monetary değeri birbirine oldukça yakın
# bu değerlere baklılarak birleştirme kararı alınabilir
# satış stratejileri buna göre oluşturulabilir
# A ve B segmentlerinin clv değerleri oldukça yüksek
# Bu segmentlere daha fazla ağırlık verilmesi daha makul bir yol olabilir
# B segmentine özel indirimler uygulanabilir
# A segmentine premiun müşteri olarak özel ayrıcalıklar tanınabilir
# Örneğin, her 3 alışverişlerine özel hediye kuponu verilebilir.
# kargo maliyetleri karşılanabilir.
# A segmenti ayrıca analiz edilip kişilerin en çok alım yaptığı
# ürünler belirlenip bu ürünlere indirimler uygulanabilir

##############################################################
# Görev 4: Sonuçların Veri Tabanına Gönderilmesi
##############################################################



pd.read_sql_query("show databases;", conn)
pd.read_sql_query("show tables", conn)

pd.read_sql_query("select * from online_retail_2010_2011 limit 10", conn)

cltv_6M.head()
cltv_6M["Customer ID"] = cltv_6M["Customer ID"].astype(int)

# Write DataFrame to MySQL using the engine (connection) created above.
# cltv_m_6.to_sql(name='cltv_prediction_seda_kayademir', con=conn, if_exists='replace', index=False)
