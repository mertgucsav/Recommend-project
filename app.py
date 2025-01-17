
import numpy as np 
import wx
import wx.grid
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.decomposition import TruncatedSVD 


anime=pd.read_csv("anime.csv")   #Dosya Okuma
rating=pd.read_csv("rating.csv")

rating_df=rating.replace(to_replace=-1,value=np.nan)
anime_df=anime.replace(to_replace="Unknown",value=np.nan) 
anime_df.dropna(how="any",axis=0,inplace=True)
rating_df.dropna(how="any",axis=0,inplace=True)
anime_df=anime_df.astype({"episodes":int}) #episodes "Unknown" değerler yüzünden object olarak tanımlanmıştı bunun türünü int olarak değiştiriyorum 

anime_df=anime.dropna(axis=0,how="any")
anime_df["rating"].replace(to_replace="Unknown",value=0)
anime_df["rating"].replace(to_replace=-1,value=0)

def CollaborativeFiltering(movie):
    anime_full_data = pd.merge(rating, anime, on="anime_id")

    # 2. NaN ve Sonsuz Değerlerini Temizle
    # Rating sütunundaki -1 değerlerini NaN ile değiştir
    anime_full_data["rating_x"] = anime_full_data["rating_x"].replace(to_replace=-1, value=np.nan)

    # Sonsuz değerleri NaN ile değiştir
    anime_full_data.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 3. NaN değerlerini sıfır ile doldur
    anime_full_data.fillna(0, inplace=True)

    anime_full_data_sampled = anime_full_data.sample(frac=0.01, random_state=17)

    # 4. Rating crosstab tablosu oluştur
    rating_crosstab = anime_full_data_sampled.pivot_table(values="rating_x", index="user_id", columns="name", aggfunc="mean")

    # 5. NaN değerlerini sıfırla doldur ve transpose işlemi yap
    X = rating_crosstab.fillna(0).T  # NaN değerlerini sıfırla doldur ve transpoze et

    # 6. SVD modelini uygula
    SVD = TruncatedSVD(n_components=16, random_state=17)
    resultan_matrix = SVD.fit_transform(X)


    # elde ettiğimiz matrisin korelasyon kat sayılarını alıyoruz
    cor_mat=np.corrcoef(resultan_matrix)
    movie_names=rating_crosstab.columns 
    movie_list=list(movie_names)
    movieid=movie_list.index(movie)
    corr_movie=cor_mat[movieid]
    recom_movie=list(movie_names[(corr_movie<1.0) & (corr_movie > 0.85)])
    return anime.loc[anime['name'].isin(recom_movie)]


def ContentBasedFiltering(title):
   
    
    tfvector=TfidfVectorizer(min_df=3,  max_features=None, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3),
            stop_words = 'english') 
    anime_df["genre"]=anime_df["genre"].fillna("")
    genre_str=anime_df["genre"].str.split(",").astype(str)
    tf_matrix=tfvector.fit_transform(genre_str)
    linear_k = linear_kernel(tf_matrix, tf_matrix)
    endeks = pd.Series(anime_df.index, index=anime_df['name']).drop_duplicates() #indis numaralarını almak için 

    idx = endeks[title]

    # Benzerlik puanlarını alıyoruz ikili örnek :(id,score)
    linear_k_scores = list(enumerate(linear_k[idx]))

    # yüksek skorları tespit etmek için bir sıralama işleminden geçiriyoruz 
    linear_k_scores = sorted(linear_k_scores, key=lambda x: x[1], reverse=True)

    # burada en yüksek 10 skoru alıyorum 
    linear_k_scores = linear_k_scores[1:11]

    #  burada yukarıda ikili olarak aldımız puanların indislerini ayırıyoruz 
    anime_indices = [i[0] for i in linear_k_scores]

    # çıktı
    return anime.iloc[anime_indices]


class MyApp(wx.App):
    def OnInit(self):
        self.frame = MainFrame(None, title="Collaborative Filtering Arayüzü")
        self.frame.Show()
        return True

import wx
import wx.grid

class MyFrame(wx.Frame):
    def __init__(self, parent, title):
        super().__init__(parent, title=title, size=(800, 800))

        # Panel oluştur
        panel = wx.Panel(self)

        # Sizer oluştur
        sizer = wx.BoxSizer(wx.VERTICAL)

        # Giriş kutusu
        self.label = wx.StaticText(panel, label="Bir Anime giriniz:")
        self.text_ctrl = wx.TextCtrl(panel, size=(300, -1))

        # Sizer'a ekleyin
        hbox1 = wx.BoxSizer(wx.HORIZONTAL)
        hbox1.Add(self.label, flag=wx.RIGHT, border=8)
        hbox1.Add(self.text_ctrl, proportion=1)
        sizer.Add(hbox1, flag=wx.EXPAND|wx.LEFT|wx.RIGHT|wx.TOP, border=10)

        # Göster düğmesi
        self.button1 = wx.Button(panel, label="ContentBasedFiltering")
        self.button1.Bind(wx.EVT_BUTTON, self.onContentBasedFiltering)

        # Sizer'a ekleyin
        sizer.Add(self.button1, flag=wx.EXPAND|wx.LEFT|wx.RIGHT|wx.TOP, border=10)

        # Göster 2 düğmesi
        self.button2 = wx.Button(panel, label="CollaborativeFiltering")
        self.button2.Bind(wx.EVT_BUTTON, self.onCollaborativeFiltering)

        # Sizer'a ekleyin
        sizer.Add(self.button2, flag=wx.EXPAND|wx.LEFT|wx.RIGHT|wx.TOP, border=10)

        # Grid (Tablo) oluştur
        self.grid = wx.grid.Grid(panel)
        self.grid.CreateGrid(10, 7)  # 5 satır, 7 sütun (ilk 5 veri için)

        # Sütun başlıklarını ayarla
        column_names = ["anime_id", "name", "genre", "type", "episodes", "rating", "members"]
        for col, name in enumerate(column_names):
            self.grid.SetColLabelValue(col, name)

        # Grid'i sizer'a ekleyin
        sizer.Add(self.grid, proportion=1, flag=wx.EXPAND|wx.ALL, border=10)

        # Panel'in sizer'ını ayarla
        panel.SetSizer(sizer)

    def onContentBasedFiltering(self, event):
        # Kullanıcının girdiği metni al
        user_input = self.text_ctrl.GetValue()

        # ContentBasedFiltering fonksiyonunu çağır
        try:
            # Burada ContentBasedFiltering fonksiyonunu çağıran kodu ekleyin
            df = ContentBasedFiltering(user_input)
            df = df.head(10)  # İlk 10 satırı al

            # Grid'e verileri yaz
            for row in range(min(len(df), 10)):
                for col, value in enumerate(df.iloc[row]):
                    self.grid.SetCellValue(row, col, str(value))

        except Exception as e:
            wx.MessageBox(f"Hata: {str(e)}", "Hata", wx.OK | wx.ICON_ERROR)

    def onCollaborativeFiltering(self, event):
        # Kullanıcının girdiği metni al
        user_input = self.text_ctrl.GetValue()

        # CollaborativeFiltering fonksiyonunu çağır
        try:
            # Burada CollaborativeFiltering fonksiyonunu çağıran kodu ekleyin
            df = CollaborativeFiltering(user_input)
            df = df.head(10)  # İlk 10 satırı al

            # Grid'e verileri yaz
            for row in range(min(len(df), 10)):
                for col, value in enumerate(df.iloc[row]):
                    self.grid.SetCellValue(row, col, str(value))

        except Exception as e:
            wx.MessageBox(f"Hata: {str(e)}", "Hata", wx.OK | wx.ICON_ERROR)

# Uygulamanın çalıştırılması
app = wx.App(False)
frame = MyFrame(None, "Anime öneri sistemi")
frame.Show()
app.MainLoop()
