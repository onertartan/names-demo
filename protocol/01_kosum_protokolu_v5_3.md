# Koşum protokolü v5.3 — final freeze adayı

`run_matrix_v4.csv` ile birlikte kullanılır (tek manifest; §Manifest).

**v5.2→v5.3 farkları** (dördüncü tur: Qwen değerlendirmesi + karar raporu; 20
zorunlu madde işlendi, ikisi karar raporunun Qwen'i düzelttiği biçimde alındı):
k̂ seçimi için yön/bağ/non-finite politikası (`k_hat_tie`, `cvi_failure` ≠
`algorithm_failure`); QC iki kestirimciye ayrıldı — `sigma_generator_deviation`
(üretici sadakati) ve `sigma_achieved` (örneklem-prototip artık sapması,
SSA-karşılaştırılabilir; transfer duyarlılığı yalnız bunu kullanır) +
`rho_max_achieved`/`rho_max_pair`; birincil Friedman, birincil/geçiş GLMM ve
AR-only Friedman **yalnız Blok A'nın 600 hücresinde** tanımlı; GLMM gözlem düzeyi
tohum-düzeyi ikili `correct` olarak açıkça yazıldı; GLMM yedek zinciri 4 aşamalı
deterministik sıraya bağlandı (Bayes/regülarize model 4-CVI frequentist'ten
SONRA); Friedman'a `N_all_tied`/`N_informative` raporu; AR-only Friedman ayrı
secondary Holm ailesi; SSA deployment boru hattı donduruldu (aynı CVI kodu, aynı
k aralığı, aynı bağ kuralı; sınır-k uyarısı; extrapolation-limited dili); geometri
kapsama birincil kestirimcisi winner'dan bağımsız average-linkage prototipleri;
`bias`/`ari_at_ktrue`/`corr_ari` kesin tanımlandı (corr_ari: yön-standardize +
Spearman, betimsel); B7 jitter seri-bazlı iid olarak açık yazıldı; aykırı
loglama alanları; tüm bloklarda son dönüşüm = satır z-norm `ddof=0` genel kuralı;
koşum altyapısı/bütünlük bölümü (benchmark, checkpoint, "runtime gerekçesiyle
parametre değişikliği yasak"); B11 tek-konfigürasyon dil sınırı; Blok D
eşit-boyut sınırlaması; configuration-düzeyi rastgele etki önerisi REDDEDİLDİ
(sabit faktöriyel tasarım), LOCO kararlılığı opsiyonel ek olarak adlandırıldı.

**v5.1→v5.2 farkları** (üçüncü tur değerlendirme; tamamı kabul edildi): B0 için
ayrı RNG anahtarı `seed_key_hucre` (kimlik/tekrar-üretim çelişkisi çözüldü); AR(1)
başlatma cümlesi düzeltildi (yasak olan `x_0=0` ve *ölçekli*-innovation başlatma;
ölçeksiz `x_0=ε_0` zaten durağandır); z-norm `ddof=0` açıkça donduruldu; RNG
"bağımsızlık garantisi" dili pratik-bağımsız akış diline çekildi; aykırı şiddeti
mutlak `σ_out=1.5` olarak sabitlendi (B6=B11) ve "yönü rastgele" iddiası nicel SNR
beklentisiyle değiştirildi (beklenen prototip korelasyonu ≈ 0.55); all-invalid
politikası `correct=0` + `algorithm_failure` olarak değiştirildi (payda 100 kalır);
GLMM aile-içi kontrastlar için "SE'ler muhafazakâr" iddiası kaldırıldı; winner'a
achieved-σ transfer duyarlılığı eklendi; B11 dili "gerçekçi" → "olumsuz birleşik
stres".

v4→v5.1 farkları, ikinci tur metodolojik değerlendirme + silhouette-cosine
tartışmasından:

1. **sil_cos yeniden sınıflandırıldı:** registry'de kalır, birincil Friedman'dan ve
   winner havuzundan çıkar; küre özdeşliği gerekçesi eklendi (§Birincil test).
2. **Birincil formülasyon seti 4'e indi** (sil_euc · DB · CH · Dunn d1/D1); winner
   havuzu 4×4=16 çift olarak donduruldu — 8-mi-5-mi belirsizliği kapandı.
3. **Blok B tamamen sayısallaştırıldı:** küme-boyutu vektörleri, σ vektörleri,
   counterbalance atamaları, aykırı-değer mekanizması, B11 bileşimi (§Blok B).
4. **Tek makine-okunur manifest:** A–D blokları `run_matrix_v4.csv`'de (§Manifest).
5. **Seed mimarisi `SeedSequence` anahtarlı:** hücreler-arası pratik-bağımsız
   RNG akışları (§Koşum).
6. **GMM, dejenere-bölütleme ve singleton politikaları donduruldu** (§Koşum).
7. **σ₅₀ köşe-durum kuralları ve marjinalleştirme profili donduruldu** (§GLMM).
8. **Winner farkı SE'si Monte Carlo belirsizliği olarak tanımlandı;** winner's-curse
   kabulü eklendi; bilgi üretmeyen hücre-sayısı-ağırlığı kontrolü kaldırıldı,
   empirical-geometry ağırlıklaması gerekçeli reddedildi (§Winner rule).
9. **Blok D: 16 koşum; prototip kaynağı average-linkage ile winner'dan bağımsız
   önceden sabitlendi;** prototip seti dosya+hash olarak dondurulur (§Blok D).
10. **Winner bandı nominal σ üzerinde tanımlı** — gerekçesiyle (§Winner rule).
11. **Silhouette agreement analizi betimsel katman olarak donduruldu** (§Analiz).
12. B7 mekanizma dili daraltıldı: "amplitude-induced SNR heterogeneity".
13. **GLMM 8 seviyeli kalır;** yakınsama başarısızlığında 4-CVI'lık ön-kayıtlı
    yedek model (§GLMM).

**Amaç:** Etiketleri bilinen sentetik veride hangi (algoritma, indeks) çiftinin doğru
$k$'yi geri kazandığını bulmak; kazananı sonraki aşamada 1880–2025 SSA verisine
uygulamak.

## Sabitler

```
Zaman ekseni   : 1880–2025, T = 146
n_per_cluster  : 10        (Blok A'da sabit; Blok B'de 5/10/20)
Aday k aralığı : 2–10
Algoritmalar   : KMeans · Ward · Complete · GMM(diag)
Registry       : sil_euc · sil_cos · DB · CH · Dunn d1/D1 · d2/D2 · d4/D1 · d3/D2
                 (8 indeks her koşumda hesaplanır ve kaydedilir)
Birincil set   : sil_euc · DB · CH · Dunn d1/D1
                 (birincil Friedman + winner havuzu + geçiş filtresi bu 4'ü kullanır;
                  sil_cos ve Dunn d2/D2 · d4/D1 · d3/D2 exploratory statüde)
Son dönüşüm    : TÜM bloklarda (A–D) kümelemeden önceki son seri-düzeyi işlem
                 satır-bazlı z-normalizasyon, ddof=0 — sil_euc/sil_cos küre
                 özdeşliğinin ve σ karşılaştırılabilirliğinin ön koşulu
```

Blok A bilinçli olarak **küçük örneklem rejiminde** çalışır: n = k×10 ≤ 80 ≪ T = 146.
Bu bir kısıt değil eşleşme — gerçek veri de aynı rejimde (39–57 tam seri, küme
başına 2–25 üye). Sonuçlar n ≫ T rejimine genellenmez; o rejim Blok B'nin
n_per_cluster taramasıyla yalnız yerel olarak yoklanır.

## Blok A — haritalama (ana blok)

| Faktör | Seviyeler |
|---|---|
| Kol | `P_konum` · `M_karisik` |
| $k_{true}$ | 3, 4, 5, 6, 8 |
| ρ hedefi | 0.456 (boşluk 10 yıl) · 0.615 (8) · 0.765 (6) |
| σ | 0.1 – 1.0, adım 0.1 |
| Gürültü yapısı | beyaz Gauss · AR(1) φ=0.97 |

30 yapılandırma × 10 σ × 2 gürültü = **600 hücre**, × 4 algoritma = 2400 koşum.
Zorluk dağılımı (verdict beyaz-gürültü geometrisinden): 2 × (80 kolay / 110 orta /
110 zor).

**Gürültü yapısı ana faktör, duyarlılık maddesi değil.** Ön ölçüm sıralamanın gürültü
yapısıyla tersine döndüğünü gösterdi: σ=0.7, ρ=0.615'te Dunn d1/D1 beyaz gürültüde
%100, AR(1) φ=0.97'de %40; silhouette (euc) %0'dan %47'ye çıkıyor. Gerçek artıkların
lag-1 otokorelasyonu medyan 0.98 (erkek) / 0.98 (kadın) — gerçek veri AR tarafında.
Nihai (algoritma, indeks) seçimi **AR(1) sütununa göre** yapılır; beyaz gürültü
sütunu literatürle karşılaştırılabilirlik ve mekanizma ayrıştırması için koşulur.

AR(1) üretimi birim varyanslı olmalı: `x_t = φ·x_{t-1} + √(1−φ²)·ε_t`. Aksi halde
σ etiketi iki gürültü yapısı arasında karşılaştırılamaz. Mekanizma notu (beklenti
olarak kayda geçir): AR gürültüsü pürüzsüzdür, tek bir seri şans eseri prototipe
benzer "hayalet şekil" üretebilir — Dunn d1/D1'in tek-nokta-çifti mimarisi buna
savunmasızdır; aynı pürüzsüzlük etkin boyutu düşürüp mesafe yoğunlaşmasını
azalttığı için silhouette kısmen toparlar.

**φ neden tek seviye (0.97):** ölçülen gerçek artık AC ≈ 0.98'in karşılığı. Ön
tarama (σ=0.7, ρ=0.615, 15 tohum) iki mekanizmanın farklı φ eşiklerinde devreye
girdiğini gösterdi: Dunn'un çöküşü φ=0.8'de tamamlanıyor (1.00→0.33), silhouette
toparlaması ancak φ≥0.95'te başlıyor (0.00→0.47 @0.97). İki-mekanizma ayrımı
Blok B'deki φ taramasıyla (B8–B10) belgelenir.

ρ seviyeleri bilinçli olarak yüksek: ön ölçüm, ρ ≤ 0.24'te bütün indekslerin σ = 1.2'ye
kadar kusursuz kaldığını gösterdi — o bölge bilgi üretmiyor. Üç seviye de geçişin
gerçekleştiği bantta.

**ρ hedefi her iki kolda ρ_max (en yakın prototip çifti) üzerinden tanımlanır.**
`M_karisik`'te bu çift daima komşu iki tepedir; karma eklerin (level_shift, trough,
cylinder) tüm çiftleri ρ < 0.34'te kalır (k=5 ölçümü: 1.–2. çift tam hedefte,
3. çift ≤ 0.34, medyan ≤ 0.16). Kollar ρ_max'te eşleşir ama tam korelasyon
dağılımında ayrışır — bu fark tasarımsaldır ve `rho_mean` / `d_eff`
eşdeğişkenleriyle izlenir.

**σ ve ρ ayrı faktörler; `ratio` tasarım ekseni DEĞİL.** Sabit-ratio testi, aynı
ratio'da silhouette'in %100'den %0'a düştüğünü gösterdi. CSV'de eşdeğişken olarak
duruyor; analizde regresör olarak kullan, hücre eşleştirmede kullanma.

## Koşum

Tek aşama: **her hücre 100 tohum.** Düz %100/%0 bölgelerinde bir kısmı bilgi
üretmeyecek; bu bilinçli bir sadelik tercihi.

**AR(1) durağan başlatma:** `x_0 ~ N(0,1)` doğrudan çekilir; sonrası
`x_t = φ·x_{t-1} + √(1−φ²)·ε_t`. Yasak olan iki başlatma: `x_0 = 0` ve **ölçekli**
innovation'la başlatma `x_0 = √(1−φ²)·ε_0` — ikincisinde Var(x_0) = 1−φ²
(φ=0.97'de 0.06) olur ve ilk onlarca zaman noktası nominal varyansın altında kalır;
T=146'da ihmal edilemez. Buna karşılık *ölçeksiz* `x_0 = ε_0` (ε_0 ~ N(0,1),
bağımsız) durağan başlatmanın ta kendisidir — v4'teki "x_0=ε_0 de hatalı" ifadesi
düzeltilmiştir. Ön-ölçüm kodunun bu iki formdan hangisini kullandığı koşum öncesi
doğrulanır ve tarihsel not buna göre kesinleştirilir (kontrol listesi).

### Seed mimarisi (v5.1'de yeniden yazıldı; v5.2'de `seed_key_hucre` eklendi)

İki gereklilik birlikte sağlanır: (i) aynı `(hücre, tohum)` sentetik veri seti dört
algoritmaya da **aynen** verilir — eşleştirilmiş karşılaştırmanın ön koşulu;
(ii) farklı hücrelere ayrı `SeedSequence` anahtarları atanır, böylece gerçekleşmeler
**pratik olarak bağımsız** akışlardan gelir (akış çakışması olasılığı ihmal
edilebilir; matematiksel garanti iddia edilmez). `seed_data = 1..100`'ün her hücrede
RNG'yi aynı biçimde başlatması yasak — aksi halde farklı σ/ρ hücreleri aynı temel
Gauss innovation'larını paylaşır ve tasarım noktaları arasında gereksiz bağımlılık
doğar.

```
PROTO_TAG = 20260810                       # protokol sürüm sabiti, değişmez
root      = np.random.SeedSequence([PROTO_TAG, seed_key_hucre, tohum])
data_ss, alg_ss = root.spawn(2)
rng_data  = np.random.default_rng(data_ss)  # TÜM veri-tarafı rastgelelik:
                                            # gürültü, jitter, aykırı-indeks seçimi
kmeans_seed, gmm_seed = [int(s.generate_state(1)[0]) for s in alg_ss.spawn(2)]
```

`seed_key_hucre` RNG anahtarıdır ve manifest sütunudur: her satırda
`seed_key_hucre = hucre_id`, tek istisna B0 (aşağıda). `hucre_id` benzersiz satır
kimliği olarak kalır (§Manifest); `tohum = 0..99`. Aynı `(seed_key_hucre, tohum)`
için üretilen X dizisi dört algoritmaya değişmeden verilir.
Ward/Complete deterministiktir, seed almaz. `seed_data ≠ seed_alg` ayrımı yapı
gereği korunur. n_init yeterince yüksek tutulur (KMeans 50, GMM 5); başlatma
değişkenliği ayrıca kaydedilmez, n_init ile bastırılır.

### Algoritma parametreleri (donduruldu)

```
KMeans : n_init=50, init='k-means++', max_iter=300, tol=1e-4, algorithm='lloyd',
         random_state=kmeans_seed
Ward   : AgglomerativeClustering(linkage='ward', metric='euclidean')
Complete: AgglomerativeClustering(linkage='complete', metric='euclidean')
GMM    : GaussianMixture(covariance_type='diag', reg_covar=1e-6, tol=1e-3,
         max_iter=500, n_init=5, init_params='kmeans', random_state=gmm_seed)
```

GMM yakınsama başarısızlığı: n_init içindeki en iyi alt-sınır çözümü kabul edilir,
`converged` bayrağı kaydedilir, hücre başına oran raporlanır; yeniden koşum yok.

### Geçersiz bölütleme politikası (tüm CVI'lar için ortak)

Bir aday k'de gerçekleşen benzersiz etiket sayısı k'den küçükse (`len(unique(labels))
< k` — pratikte yalnız GMM'de mümkün; KMeans boş kümeyi yeniden konumlandırır,
bağlantı yöntemleri tam k üretir), o aday **o (hücre, tohum, algoritma) için
geçersizdir**: sekiz CVI da onu atlar, k̂ argmax'ı geçerli adaylar üzerinden alınır.
Dejenere-aday oranı hücre başına kaydedilir. Bütün adaylar geçersizse (beklenen
sıklık ~0): `k_hat = NA`, **`correct = 0`**, `algorithm_failure = 1` — payda 100
tohum olarak kalır. Gerekçe: hedef nicelik (algoritma, CVI) çiftinin
güvenilirliğidir; geçerli bölütleme üretememek yöntem başarısızlığıdır ve bu
gözlemleri paydadan çıkarmak özellikle GMM accuracy'sini yukarı yanlı kılar.
`bias` ve `ari_at_ktrue` bu gözlemlerde NA olur ve yalnız o destekleyici
özetlerden düşer; başarısızlık oranı ayrıca raporlanır. Kural sekiz CVI için
ortaktır ve CVI-spesifik olamaz — geçerlilik bölütlemenin özelliğidir, indeksin
değil.

### Singleton politikası

Tek-üyeli küme **geçerli** bölütlemedir, dejenerelik değil. CVI davranışları
donduruldu: silhouette'ta singleton gözlem için s(i)=0 (Rousseeuw konvansiyonu);
Dunn varyantlarında singleton çapı Δ=0 (D1 ve D2 tanımlarının ikisinde de — boş
çift kümesi üzerinden maksimum/ortalama 0 olarak tanımlanır); DB'de S_i=0; CH
etkilenmez. Registry kodunun bu tanımları uyguladığı koşum öncesi testle doğrulanır
(koşum-öncesi kontrol listesi maddesi).

### k̂ seçimi: yön, bağ ve non-finite politikası (donduruldu)

Her CVI için k̂ yalnız **geçerli** adaylar üzerinden seçilir. Yönler: silhouette,
CH ve dört Dunn varyantı maksimize; **DB minimize** edilir. Bağ (tie) kümesi en
iyi skora göre tanımlanır (ikili karşılaştırma zincirinin geçişkenlik sorunundan
kaçınmak için): S* = yöne göre en iyi skor olmak üzere,
`T = {k geçerli : np.isclose(S(k), S*, rtol=1e-10, atol=1e-12)}`;
**k̂ = min(T)** (parsimoni konvansiyonu), `|T| > 1` ise `k_hat_tie = 1` kaydedilir.
Ölçek-farkındalıklı `isclose` kullanılır — CH gibi büyük-ölçekli ve silhouette
gibi [−1,1]-ölçekli indeksler tek mutlak eşikle karşılaştırılamaz.

Geçerli bir bölütleme üzerinde bir CVI non-finite (NaN/Inf) değer üretirse o aday
**yalnız o CVI için** dışlanır. Bir CVI için hiç finite aday kalmazsa:
`k_hat = NA`, `correct = 0`, `cvi_failure = 1` — bu `algorithm_failure` ile
karıştırılmaz (`algorithm_failure` bölütleme üretilememesi, `cvi_failure` indeksin
hesaplanamamasıdır; ikisi ayrı alanlardır ve ayrı raporlanır). Aynı yön/bağ/
non-finite kuralları SSA deployment aşamasında değişmeden kullanılır (§SSA).

### Gerçekleşen geometri QC (v5.3'te iki kestirimciye ayrıldı)

İki ayrı nicelik kaydedilir; rolleri farklıdır ve karıştırılmaz:

**1. `sigma_generator_deviation` (üretici sadakati):** serinin *nominal üretici
prototipe* göre artık sapması (z-norm sonrası, ddof=0). Yalnız üretim boru
hattının doğruluğunu izler.

**2. `sigma_achieved` (transfer-karşılaştırılabilir):** her gerçekleşmede
gerçek etiketlerle örneklem sınıf prototipi `P̂_c = z(mean_{i∈c} z_i)` hesaplanır;
`sigma_achieved = SD_ddof0(z_i − P̂_{c(i)})`. Gerekçe: SSA'daki ampirik band
(0.35–0.81) gerçek üretici prototip bilinmeden, *kestirilen* küme prototipleri
çevresindeki artık sapma olarak ölçüldü — sentetik tarafta aynı kestirimci
ailesi kullanılmazsa band karşılaştırması elma-armut olur. **SSA bandıyla ve
winner'ın achieved-σ duyarlılığıyla yalnız `sigma_achieved` karşılaştırılır.**
Bilinen ikinci-derece fark (kayıtlı sınırlama): sentetik tarafta prototipler
gerçek etiketle, SSA tarafında kümeleme etiketiyle kuruludur; zor rejimlerde
kümeleme-etiketli artık sapma bir miktar küçük ölçer. Bu farkın ölçülü sınırı,
`sigma_hat`'in beş yöntem arasında ≤ %17 oynaması (§Transfer).

**3. `rho_max_achieved`:** aynı örneklem prototipleri `P̂_c` arasında maksimum
korelasyon; en yüksek korelasyonu veren çift `rho_max_pair` alanına yazılır.

Hücre özeti: tohumlar üzerinden **medyan ve IQR**. Nominal-gerçekleşen sapması AR
hücrelerinde beyazdan büyük olacaktır (pürüzsüz gürültü efektif sapmayı küçültür);
bu beklenen bir olgudur ve raporlanır, düzeltilmez. Bu tablo winner bandının
nominal-σ tanımının okunma anahtarıdır (§Winner rule).

**ACF uyum kontrolü (koşum öncesi, bir kez):** gerçek artıkların ACF(1..10) eğrisi
AR(0.97)'nin kuramsal eğrisiyle karşılaştırılır. Uyum sınırlıysa makale dili
"realistic AR residual model" değil **"AR(1) calibrated to empirical lag-1
autocorrelation"** olur.

## Blok B — yerel tek-faktör duyarlılık analizi

("Robustness" değil: referans nokta çevresinde one-factor-at-a-time tarama.)
Referans yapılandırma **R** = manifestteki A hücresi (`M_karisik`, k=5, ρ=0.615,
σ=0.5, AR(1) φ=0.97, n_per=10):

```
Sınıflar (sabit sıra, indeks 1..5):
  1: peak@1944w15   2: peak@1952w15   3: peak@1960w15
  4: level_shift@1955   5: trough@1975w15
```

Counterbalance hedefleri (donduruldu, performansa değil yapıya göre seçildi):
**tepe-sınıfı hedefi = peak@1952w15** (her iki en-yakın-çiftin ortak üyesi — azami
gerilim noktası); **karma-ek hedefi = level_shift@1955** (konumu tepe bandının
içinde kalan, dolayısıyla tepelerle en iç içe karma ek). Atama A1 = manipülasyon
tepe-sınıfına, A2 = karma-ek sınıfına; kalan kümeler her zaman listelenen sabit
sırayla doldurulur.

### Koşul tablosu (tam sayısal)

| # | Koşul | Küme boyutları (sınıf 1..5) | σ vektörü (sınıf 1..5) | Diğer |
|---|---|---|---|---|
| B0 | referans | 10,10,10,10,10 (N=50) | 0.5×5 | — |
| B1 | n_per=5 | 5×5 (N=25) | 0.5×5 | — |
| B2 | n_per=20 | 20×5 (N=100) | 0.5×5 | — |
| B3a | denge 3:1, A1 | 10,**15**,10,**5**,10 (N=50) | 0.5×5 | büyük→peak@1952, küçük→level_shift |
| B3b | denge 3:1, A2 | 10,**5**,10,**15**,10 (N=50) | 0.5×5 | ayna |
| B4a | denge 5:1, A1 | 9,**20**,9,**4**,8 (N=50) | 0.5×5 | büyük→peak@1952, küçük→level_shift |
| B4b | denge 5:1, A2 | 9,**4**,9,**20**,8 (N=50) | 0.5×5 | ayna |
| B5a | σ 2:1, A1 | 10×5 | 0.5,**0.7**,0.5,**0.35**,0.5 | yüksek→peak@1952, düşük→level_shift |
| B5b | σ 2:1, A2 | 10×5 | 0.5,**0.35**,0.5,**0.7**,0.5 | ayna |
| B6 | %5 aykırı | 10×5 | 0.5×5 | n_out=3 (mekanizma aşağıda) |
| B7 | jitter açık | 10×5 | 0.5×5 | a~U(0.5,2.0) |
| B8 | φ=0.8 | 10×5 | 0.5×5 | AR şiddeti ↓↓ |
| B9 | φ=0.9 | 10×5 | 0.5×5 | AR şiddeti ↓ |
| B10 | φ=0.99 | 10×5 | 0.5×5 | AR şiddeti ↑ |
| B11 | kombine stres | 10,**15**,10,**5**,10 (N=50) | 0.5,**0.35**,0.5,**0.7**,0.5 | + n_out=3; hizalama aşağıda |

15 koşul satırı × 4 algoritma = **60 koşum**, 100 tohum. B0, R hücresinin B boru
hattından yeniden koşumudur — kendi benzersiz `hucre_id`'sini (612) taşır fakat
`seed_key_hucre` olarak R hücresinin kimliğini alır; böylece aynı
`(seed_key, tohum)` akışıyla A sonucunu birebir yeniden üretmesi beklenir.
Ücretsiz bütünlük kontrolü: üretmezse boru hattı hatası var demektir. (v5.1'deki
"A'daki hucre_id korunur" notu manifest benzersizliğiyle çelişiyordu; `seed_key`
ayrımı çelişkiyi çözer.)

**Boyut vektörleri üzerine kayıt:** hem tam oran hem N=50 aynı anda ancak orta
kümeler 10'dan saptırılarak tutturulabilir. Öncelik sırası donduruldu: (1) N=50
sabit — B1/B2 örneklem etkisini zaten ayrı taradığı için N-confound kabul edilemez;
(2) max:min oranı tam (3.0 ve 5.0); (3) orta kümeler 10'a olabildiğince yakın.
B4'te bunun sonucu {20,9,9,8,4}'tür; orta kümelerin 9,9,8 olması dokümante edilmiş
ikincil sapmadır. B5'te seri-ağırlıklı ortalama σ = 0.51 ≈ 0.50 — manipülasyon
ortalama gürültü düzeyi değil sınıflar-arası yayılım heterojenliğidir; 0.51
gerçekleşen değer olarak kaydedilir.

**Aykırı-değer mekanizması (B6, B11 — donduruldu):** `n_out = ⌈0.05·N⌉ = 3`
(gerçekleşen oran %6, kayda geçer). Ekleme değil **yerine koyma** — N sabit kalır.
`rng_data` ile tüm N indeksinden tabakasız, yerine-koymasız 3 seri seçilir; seçilen
seri aynı prototipten, aynı etiketle, fakat **mutlak `σ_out = 1.5`** ile yeniden
üretilir (kontaminasyon modeli; referans hücrede 3·σ_ref'e denk gelir). σ_out
küme-spesifik σ'nın çarpanı DEĞİLDİR — B11'de de 1.5'tir; böylece B6 ile B11'in
kontaminasyon şiddeti özdeş kalır ve B11 yalnız bileşimi (dengesizlik +
heteroskedastisite + aynı kontaminasyon) test eder. Etiketler korunur, k_true
değişmez. Nicel beklenti (önceden kayıtlı): aykırı serinin kendi prototipiyle
beklenen korelasyonu ≈ 1/√(1+σ_out²) ≈ 0.55 — σ=0.5'lik normal serinin ≈ 0.89'una
karşı; yani "yönü rastgele" değil, **belirgin biçimde düşürülmüş SNR**. Gerçekleşen
değer QC'de raporlanır. **Loglama:** değiştirilen serilerin `outlier_series_id` ve
`outlier_class_id` alanları kaydedilir; seçim tabakasız olduğundan bazı
gerçekleşmelerde aykırıların tek sınıfta toplanması mümkündür — bu tasarım hatası
değildir, yorum aşamasında teşhis edilebilsin diye loglanır. Alternatif mekanizma
(saf gürültü serisi, etiketsiz) k_true'yu belirsizleştirdiği için reddedildi.

**B11 hizalaması (donduruldu):** boyut ve σ manipülasyonları **çapraz** bindirilir —
küçük küme (n=5, level_shift@1955) yüksek σ'yı (0.7) alır, büyük küme (n=15,
peak@1952w15) düşük σ'yı (0.35). Küçük+gürültülü küme en zorlayıcı bileşimdir;
koşulun adı **"olumsuz birleşik stres" (adverse combined-stress)** olarak konur —
SSA'da küçük-küme ↔ yüksek-σ ortak oluşumu ampirik olarak gösterilmediğinden
"gerçekçi" sıfatı kullanılmaz. Seri-ağırlıklı ortalama σ = 0.475
(kayda geçer). B11 tek atamayla koşulur (counterbalance yok — bütçe v4 ile aynı)
ve winner raporunda ikincil kanıt satırıdır. **Dil sınırı (önceden kayıtlı):** B11
tek bir olumsuz konfigürasyonu test eder; atama permütasyonları üzerinden ortalama
bir birleşik-stres etkisi KESTİRMEZ — sonuç cümleleri "a single adverse
combined-stress configuration" diye dar yazılır, "general combined-stress effect"
denmez.

**Atama counterbalance'ı:** B3–B5 iki permütasyonla koşulur, ortalama raporlanır;
permütasyonlar arası fark büyükse ayrıca not edilir (shape × koşul confound izleme).

**Jitter tanımı ve uygulama sırası (kodda tanımlı):** B7'de çarpan **seri bazında
bağımsız** çekilir — `a_i ~iid U(0.5, 2.0)`, her seri için ayrı, küme düzeyinde
değil (mevcut `make_dataset_from_prototypes` zaten böyle uygular). Sıra: genlik →
gürültü → z-normalizasyon (ddof=0).
z-norm çarpanı matematiksel olarak siler (z(a·φ)=z(φ)) ama gürültü genlikten SONRA
eklendiği için düşük-a serileri orantısal olarak daha gürültülü kalır — jitter
no-op değildir, kümeleri ışınsı biçimde uzatır. Sıra değiştirilmez. B7'nin makale
dili buna göre daraltılır: manipülasyonun mekanizması "genlik değişkenliği" değil
**"amplitude-induced SNR heterogeneity"** — saf genlik ölçeklemesi son z-norm'dan
sağ çıkmaz, etki düşük-genlik serilerin düşük SNR'sinden gelir.

Beklentiler (önceden kayıtlı): DB ve CH eşit yayılım varsayımına yakın çalışır →
B3–B5'te bozulmaları beklenir. Dunn d1/D1 B6'da çökmeli; d3/D2 dayanmalı (Bezdek ve
ark. 1998; d3/D2 exploratory statüde olduğundan bu beklenti aile-içi katmanda
raporlanır). B11'de tekil zayıflıkların bileşkesi en az tekil maksimum kadar kötü
olmalı; belirgin süper-additif bozulma varsa makalede ayrıca tartışılır.

## Blok C — sabit ratio, değişen ρ

Bu blok winner seçimine girmez; bir **zorluk-metriği falsifikasyon deneyidir**.
Sorunun testi: "aynı `ratio` DEĞERİNDE performans yapılandırmadan bağımsız mı?"
12 hücre (manifestte `C_sabit_ratio`): ratio ∈ {1.5, 2.5}, her birinde ρ 0.06'dan
0.83'e altı seviye, σ = tan(θ_min / ratio). 100 tohum, 4 algoritma = 48 koşum.

Beklenen sonuç: sabit ratio'da performans ρ ile birlikte düşer → `ratio` tek eksenli
zorluk ölçüsü olarak yetersiz. Makalede "zorluk metriklerinin sınırı" alt bölümü.

**Kapsam sınırı:** yalnız beyaz gürültüyle koşulur; bulgu beyaz-gürültü geometrisine
aittir. `σ = tan(θ_min/ratio)` kalibrasyonu izotropik gürültü varsayar; AR(1)
altında aynı σ daha küçük efektif sapma üretir (ölçüm, σ=0.7: beyaz 34.5° ≈ formülün
35.0°'si; φ=0.97'de 27.2° — formül %29 fazla tahmin). Genişletme düşük öncelikli.

## Blok D — semi-synthetic köprü

**Hücreler:** 2 cinsiyet × σ ∈ {0.4, 0.6} = 4 hücre; AR(1) φ=0.97 (durağan
başlatma); n_per_cluster=10 → N=50; aday k aralığı 2–10; 100 tohum.
**4 hücre × 4 algoritma = 16 koşum** (v4'teki "~12" düzeltildi — tablo birimi
hücre×algoritma).

**Prototip kaynağı (dairesellik önlemi — koşumdan önce dondurulur):** ampirik
prototipler **average-linkage** (Euclidean, z-normalize seriler, k=5) ile çıkarılır.
Seçim gerekçesi: (i) dört yarışmacıdan biri değil — kazanan hangi çift olursa olsun
kendi ürettiği geometride yeniden test edilmiş olmaz; (ii) `sigma_hat` ölçümünde
zaten kullanılan beş yöntemden biri, yani veri boru hattında mevcut. Prototip =
küme üyelerinin ortalamasının z-normalizesi; tek-üyeli küme çıkarsa prototip o
serinin kendisidir (kaydedilir). İki cinsiyet için iki prototip seti, ana koşum
başlamadan `.npy` olarak kaydedilir; `prototype_set_id` + SHA256 hash manifeste ve
protokol ekine girer. Ground-truth etiketler üretim gereği bilinir (prototip i'den
üretilen seri i etiketlidir).

**QC:** `sigma_achieved` ve ampirik prototipler arası `rho_max_achieved` kaydedilir;
ampirik `rho_max` / `rho_mean` / `d_eff` değerlerinin Blok A kapsama bölgesine
düşüp düşmediği geometri kapsama kontrolüne (§Transfer) beslenir. **Kapsam sınırı
(önceden kayıtlı):** Blok D ampirik prototip *şekil* geometrisini taşır, ampirik
küme-boyutu dağılımını taşımaz — eşit boyut (n_per=10) kullanılır; boyut
dengesizliği etkileri ayrı olarak Blok B'de değerlendirilir ve D sonuçları buna
göre dar yorumlanır.

Hedef sonuç dili: "the selected pair was the most reliable under both controlled
shape-based simulations calibrated to the empirical regime and a semi-synthetic
empirical-prototype stress test." Kazanan Blok D'de ciddi bozuluyorsa bu saklanmaz —
transfer varsayımının sınırı olarak raporlanır.

## Manifest — `run_matrix_v4.csv`

Protokol metni ile koşum kodu arasında ikinci bir tasarım kaynağı bırakılmaz: dört
blok tek makine-okunur dosyada. v3'ün 612 satırı mevcut sütunlarıyla **bit-değişmez**
taşınır (`hucre_id` 0–611 dosya sırasıyla dondurulur); yeni sütunlar eklenir, B
satırları 612–626, D satırları 627–630 olarak eklenir. Toplam 631 satır.

```
Sütunlar (ortak süperset; ilgisiz alan boş bırakılır, boş = referans değer):
blok            A_harita | B_duyarlilik | C_sabit_ratio | D_koprusu
hucre_id        0..630 (dondurulmuş, benzersiz)
seed_key_hucre  RNG anahtarı; her satırda = hucre_id, tek istisna B0'da = R'nin id'si
kol             P_konum | M_karisik            (A)
k_true          3|4|5|6|8                      (A; B/D'de 5)
rho_hedef       r045|r062|r078                 (A/B)
sigma           nominal σ (B5/B11'de seri-ağırlıklı ortalama; vektör ayrı sütunda)
gurultu         beyaz | ar1_080 | ar1_090 | ar1_097 | ar1_099
phi             0.8|0.9|0.97|0.99              (AR satırlarında)
n_per_cluster   5|10|20
kume_boyutlari  ';' ayraçlı 5 tamsayı, sınıf sırası 1..5   (B3/B4/B11)
sigma_vektoru   ';' ayraçlı 5 ondalık, sınıf sırası 1..5   (B5/B11)
atama_id        A1|A2                          (B3–B5)
aykiri_n        3                              (B6/B11)
aykiri_sigma    1.5  (mutlak; küme σ'sından bağımsız)  (B6/B11)
jitter          0|1                            (B7'de 1)
cinsiyet        erkek|kadin                    (D)
prototip_set_id + prototip_hash                (D)
siniflar        '|' ayraçlı sınıf listesi      (A/B/C; D'de prototip dosyasına atıf)
rho_max, theta_min, ratio, verdict, rho_mean, d_eff   (önceden hesaplanmış geometri;
                B satırlarında R hücresinin değerleri, D'de QC sonrası doldurulur)
```

Sürücü betiği bu dosyayı okur, `TimeSeriesSyntheticDataGenerator` +
`optimal_k_analysis` çağırır, düz kayıt tablosu yazar. Arayüz keşif içindir,
matris için değil.

## Kayıt

Her (hücre, algoritma, indeks, tohum) için: `k_hat`, `correct`, `bias`, `corr_ari`,
`k_hat_tie`, `cvi_failure`; (hücre, algoritma, tohum) düzeyinde: `ari_at_ktrue`,
`degenerate_candidates` (hangi k'ler geçersizdi), `algorithm_failure`, GMM
`converged`, `outlier_series_id`/`outlier_class_id` (B6/B11); hücre düzeyinde:
`sigma_generator_deviation`, `sigma_achieved`, `rho_max_achieved`, `rho_max_pair`
(medyan + IQR). Kesin tanımlar (donduruldu):

- **`bias` = k̂ − k_true** (işaretli); `k_hat = NA` ise `bias = NA` ve yalnız
  bias özetlerinden düşer.
- **`ari_at_ktrue` = ARI(y_true, ŷ_{algoritma, k=k_true})** — algoritmaya doğru
  k verildiğinde bölütlemeyi ne kadar doğru bulduğunu ölçer; k-seçim başarısından
  ayrıdır ve **CVI'dan bağımsızdır** (sekiz CVI satırında aynı değer; algoritma
  düzeyinde bir alandır).
- **`corr_ari`** (tutuldu; betimsel-yalnız, asla test edilmez): önce yön
  standardize edilir — `S*_DB(k) = −DB(k)`, diğerlerinde doğal yön — sonra
  geçerli ve finite aday k'lar üzerinde **Spearman(S*(k), ARI(k))**. Amaç lineer
  ölçek uyumu değil aday-k sıralama uyumu olduğundan Spearman. Geçerli-finite
  aday sayısı < 3 ise `corr_ari = NA`. Tutulma gerekçesi: registry'de CVI
  *eğrisini* (argmax'ı değil) bölütleme kalitesine bağlayan tek alan; mekanizma
  anlatısı (örn. AR altında Dunn'un hayalet-şekil çöküşü) için ucuz teşhis.
  Çıkarma seçeneği kayda geçti; karar "tut" (onay listesi).

## Analiz

Üç katman, her birinin rolü farklı. Sıra önemli: birincil test önce, etki tahmini
sonra, görsel her ikisinin özeti.

### 1. Birincil — Friedman + Nemenyi (hipotez testi)

Hücre içi toplama önce: 100 tohumun `correct` oranı → hücre başına tek doğruluk
sayısı; `bias` ve `corr_ari` için ortalama. Tohum başına test yok — n'i yapay şişirir.

Bütün indeksler aynı bölütlemeler üzerinde puanlandığı için gözlemler eşleştirilmiş;
Friedman, anlamlıysa Nemenyi post-hoc ve kritik fark diyagramı (Demšar 2006).

**Test yapısı ve Holm kapsamı:** dört algoritma için dört ayrı Friedman omnibus
testi; bu dört p-değerine Holm. Holm sonrası anlamlı algoritmalarda **4 birincil
formülasyon** üzerinde Nemenyi. Aile-içi katman ayrı bir çıkarımsal aile: algoritma
başına Dunn-içi Friedman → Holm → anlamlıysa Nemenyi.

**Bağ (tie) işleme:** kolay/zor uçlarda çok sayıda özdeş accuracy beklenir.
Friedman **mid-ranks + tie correction** ile; kritik fark diyagramının yanında
accuracy farkları (büyüklük) da raporlanır — sıra analizi 0.51-0.50 ile 1.00-0.00
farklarını aynı görür.

**Yorum sınırı:** 600 hücre bağımsız 600 gerçek veri kümesi değil, ön-tanımlı
faktöriyel tasarım noktalarıdır; sonuçlar "across the prespecified simulation
design" diliyle sınırlandırılır.

**AR-only secondary (ön-tanımlı):** birincil Friedman Blok A'nın 600 hücresinde;
ek olarak yalnız Blok A'nın 300 AR(1) hücresinde ikinci bir Friedman. Sonuca-göre-
seçilmiş bir alt küme değil — SSA kalibrasyonuna göre önceden belirlenmiş
hedef-alan analizi; winner rule popülasyonuyla tutarlı. **Çoklu test yapısı:**
dört algoritmanın AR-only omnibus p-değerleri birincil Holm ailesine
KARIŞTIRILMAZ; kendi **ayrı secondary Holm ailesi** içinde düzeltilir, Holm
sonrası anlamlı algoritmalarda 4 birincil formülasyon üzerinde Nemenyi.

**İnformatif hücre raporu:** kolay/zor uçlarda çok sayıda tam bağ beklenir; her
algoritma için birincil ve AR-only Friedman raporlarında `N_total`, `N_all_tied`
(dört formülasyonun hücre-accuracy'lerinin özdeş olduğu hücre sayısı) ve
`N_informative = N_total − N_all_tied` verilir — testin fiilen hangi hücrelerden
güç aldığı görünür olur.

**Birincil test 4 formülasyon üzerinde: sil_euc · DB · CH · Dunn d1/D1.**
Aile-temsilcisi kuralı artık iki aileye de simetrik uygulanır:

- *Dunn ailesi → d1/D1.* Dört varyant birbirine, diğer indekslerin birbirine
  olduğundan çok daha bağımlıdır; sekiz yarışmacı sıralama uzayını Dunn ailesiyle
  doldurur, k artınca Nemenyi kritik farkı büyür. Temsilci performansa göre değil
  konvansiyona göre seçildi (literatürün "Dunn indeksi" dediği tanım).
- *Silhouette ailesi → sil_euc.* Gerekçe küre özdeşliğidir: satır bazında
  z-normalize serilerde her serinin normu tam √T'dir (**z-norm `ddof=0` — popülasyon
  std — ile; donduruldu.** `ddof=1` normu √(T−1) yapıp özdeşliğin katsayısını
  değiştirir; mevcut `shapes.zscore` zaten ddof=0 uygular), dolayısıyla her çift için
  `d_E² = 2T·(1−ρ) = 2T·d_cos` — Euclidean ve cosine uzaklıkları aynı korelasyon
  geometrisinin monoton dönüşümleridir ve cosine burada Pearson korelasyonuna eşittir.
  İki silhouette'in ayrışabildiği tek kanal, silhouette'in uzaklık *ortalamaları*
  alması ve karekök dönüşümünün doğrusal olmamasıdır (konkav yeniden ölçekleme).
  Bu, iki bağımsız geometrik ilke değil aynı geometrinin iki temsilcisidir; Dunn'a
  uygulanan kural gereği tek temsilci yarışır. Temsilci sil_euc'tur — dört
  algoritmayla ortak Euclidean değerlendirme geometrisi kurduğu için (metric-
  congruence; GMM(diag) saf Euclidean olmadığından "dört algoritma da Euclid'le
  eğitiliyor" denmez, dil "a common Euclidean evaluation geometry across clustering
  methods" olur). Seçim performansa göre yapılmadı.

**Dil sınırları:** birincil sonuç cümleleri temsilciye dar yazılır — "Dunn family
underperformed" değil "the conventional d1/D1 formulation underperformed";
"Silhouette failed" değil "the Euclidean Silhouette formulation failed". Exploratory
katmanda sil_cos sil_euc'tan iyi görünürse yorum "cosine daha iyi metrik" DEĞİL,
"silhouette'in k-seçimi uzaklıkların konkav dönüşümüne duyarlı" şeklindedir —
Blok C'nin "aynı ratio ≠ aynı zorluk" dil disipliniyle aynı ruhta.

**Aile içi Friedman (ayrı katman, yalnız Dunn):** dört Dunn varyantı kendi
aralarında ayrı Friedman + Nemenyi. Silhouette çifti için çıkarımsal katman
AÇILMAZ — iki üyeli ailede agreement analizi yeterlidir (aşağıda). Koşum tarafında
hiçbir şey değişmez: registry sekiz indeksi her hücrede hesaplar; ayrım yalnız
test yapısındadır.

**Silhouette agreement analizi (betimsel — donduruldu):** birim (algoritma, hücre,
tohum); ölçü `P(k̂_sil_euc = k̂_sil_cos)`; genel oran + gürültü×σ×ρ katmanlı tablo +
**"en az biri yanlışken uyuşma"** koşullu oranı. Hiçbir çıkarımsal aileye girmez,
test edilmez. Okuma uyarısı önceden kayıtlı: düz %100 bölgelerinde iki formülasyon
da doğru k'yi seçtiği için genel uyuşma otomatik şişer; bilgi katmanlı tabloda ve
koşullu orandadır. Uyuşma yüksekse çıkarma kararının ampirik desteği; zor rejimlerde
düşükse konkav-dönüşüm duyarlılığı ayrı bir mekanizma bulgusu olarak raporlanır.

Birincil değişken `correct`; `bias` ve `corr_ari` destekleyici (ayrı test edilirse
ailesel hata şişer — destekleyici olarak raporla, test etme).

Bu katman **Blok A'nın 600 hücresinde** koşulur — düz %100/%0 hücreler çöp değil
bilgidir.
Sonuca bağlı bir filtre hiçbir birincil analize uygulanmaz; geçiş-bandına kısıtlı
analizler yalnız ayrı estimand'lı secondary katmanlarda yaşar.

### 2. İkincil — karma etkili lojistik regresyon (etki tahmini)

**Algoritma başına ayrı GLMM:**

```
Algoritma a için:
logit P(correct) = CVI + σc + σc² + ρ_f + k_f + gürültü + kol
                 + CVI×σc + CVI×σc² + CVI×gürültü + CVI×kol
                 + (1 | hücre) + (1 | hücre:tohum)
```

Kodlama kararları:
- **CVI faktörü 8 seviyeli kalır** (sil_cos dahil). GLMM'in amacı test değil etki
  tahmini; "her formülasyon σ ile nasıl bozuluyor" sorusu hesaplanan sekiz
  formülasyonun hepsi için anlamlıdır ve tek modelde tutarlı tahmin edilir.
  **Yorum sınırı:** sil_euc↔sil_cos ve Dunn-içi kontrastlar exploratory'dir;
  confirmatory anlatıya yalnız 4 birincil formülasyonun kestirimleri girer.
  **SE dipnotu:** rastgele-kesişim yapısı ortak hücre/tohum bağımlılığını yakalar,
  fakat aile-içi (sil_euc↔sil_cos, Dunn-içi) ek artık bağımlılığı açıkça
  modellemez; bu kontrastların model-tabanlı SE'lerine — yön iddiası dahil —
  confirmatory yorum verilmez, exploratory okunurlar. (Doğrusal kuramdaki "pozitif
  kovaryansı ihmal fark-SE'sini şişirir" sezgisi, ikili GLMM + yanlış-belirlenmiş
  kovaryans + model-tabanlı SE altında garanti değildir; v5.1'deki "muhafazakâr"
  iddiası bu yüzden kaldırıldı.)
- **Ön-kayıtlı yedek zinciri (deterministik, tek-kaynak kuralı):**
  Aşama 1: 8-CVI frequentist GLMM (varsayılan optimizer). Aşama 2: yakınsama /
  tekillik / separation sorununda alternatif optimizer (**bobyqa**; ikinci
  alternatif **Nelder-Mead** — ikisi de burada donduruldu). Aşama 3: sorun
  sürerse **4 birincil CVI'lık frequentist model.** Aşama 4: 4-CVI modelde de
  ciddi yakınsama/separation sürerse **zayıf-regülarize karma model**:
  `blme::bglmer`, sabit etkilere normal(0, 2.5²) prior (standardize
  prediktörlerde), kovaryansa paket-varsayılanı gamma prior, aynı yakınsama
  ölçütleri — paket ve prior'lar koşumdan önce dondurulmuştur. Sıra gerekçesi:
  confirmatory model önce sadeleşir (exploratory CVI seviyelerini bırakmak),
  paradigma değişimi en sona kalır. Yedekler paralel model değildir; raporlanan
  her kestirimin tek kaynağı vardır ve hangi aşamanın kullanıldığı kayda geçer.
- **ρ_f ve k_f factor** (sürekli değil).
- **kol (arm) modelde;** CVI×kol geometri-farkı duyarlılığını ölçer.
- **σc merkezlenmiş; karesel form** spline yerine; uyum katman 3'te ham eğrilere
  bindirilerek doğrulanır.
- **Geometri eşdeğişkenleri (`ratio`, `rho_mean`, `d_eff`) birincil GLMM'de YOK;**
  ayrı ikincil geometri-teşhis modelinde kol yerine geçerler.
- **Rastgele yapı `(1|hücre) + (1|hücre:tohum)`;** tohum düzeyi aynı bölütlemeyi
  paylaşan sekiz CVI gözleminin bağımlılığını modeller.

**Veri kapsamı — Blok A, full-data primary:** GLMM **Blok A'nın 600 hücresiyle**
kurulur; %0/%100 hücreler dahil. Manifest birleştikten sonra "tüm hücreler"
ifadesi belirsizleşmişti — formüldeki ρ_f/k_f/kol/gürültü faktörleri yalnız Blok
A'da tanımlıdır; B/C/D kendi rollerinde ayrı analiz edilir (§Katmanlar).

**Gözlem düzeyi (açıkça donduruldu):** gözlem birimi (Blok-A hücresi, algoritma,
CVI, tohum) başına ikili `correct ∈ {0,1}` — tohum-toplanmış hücre accuracy'si
DEĞİL (rastgele `hücre:tohum` kesişimi zaten tohum-düzeyi gözlem gerektirir;
artık örtük değil açık). `algorithm_failure` gözlemleri modelde `correct = 0`
olarak kalır. Friedman ile fark netleştirildi: Friedman tohum-toplanmış hücre
accuracy'siyle, GLMM tohum-düzeyi ikili sonuçla çalışır.
Gerçek separation/yakınsama sorununda çözüm filtre değil, yukarıdaki yedek
zincirinin ilgili aşamasıdır (nihai durak: dondurulmuş zayıf-regülarize model).

**Geçiş-rejimi modeli (secondary):** 0.05–0.95 bandına kısıtlı model ayrı bir
**geçiş-rejimi estimand'ı** olarak koşulur. Filtre hücre düzeyinde ortaktır, asla
CVI-spesifik değil — hücrenin dahil olması, o hücredeki **4 birincil CVI'ın**
(ilgili algoritmada) ortalama accuracy'sinin 0.05–0.95 bandında olmasıyla belirlenir;
bir hücre girince bütün CVI gözlemleri girer (common support). Not: birincil set
4'e inince filtre ölçütü aile-dengeli hale geldi (1 silhouette + 1 Dunn + DB + CH) —
v4 değerlendirmesindeki "family-balanced transition score" endişesi kendiliğinden
çözüldü.

**Sonuç ölçüleri:** karesel modelde eğim σ'ya bağlıdır (∂η/∂σ = β₁ + 2β₂σc).
Raporlananlar: seçili σ noktalarında marjinal eğimler, predicted probability
eğrileri, ve **σ₅₀**.

**σ₅₀ tanımı (köşe durumlarıyla — donduruldu):** her (algoritma, CVI, gürültü
yapısı) için, GLMM'in predicted P(correct) eğrisi ρ_f × k_f × kol üzerinden **eşit
tasarım ağırlıklarıyla marjinalleştirilerek** hesaplanır. σ₅₀ = gözlenen σ ∈
[0.1, 1.0] aralığındaki **ilk azalan-yönlü** P=0.5 kesişimi. Kesişim aralık üstünde
→ `>1.0`; altında → `<0.1`; aralık dışına ekstrapolasyon yapılmaz; eğri monoton
değilse ve birden çok kesişim varsa ilk azalan kesişim alınır. Bu kural sonuçlar
görüldükten sonra değiştirilmez.

### Katmanların hücre setleri

Birincil Friedman ve birincil GLMM **aynı popülasyonda**: Blok A'nın 600 hücresi.
**Blok A kuralı (donduruldu):** birincil Friedman, birincil GLMM, geçiş-rejimi
GLMM ve AR-only Friedman yalnız Blok A üzerinde tanımlıdır. Blok B (yerel
duyarlılık), Blok C (zorluk-metriği falsifikasyonu) ve Blok D (semi-synthetic
transfer köprüsü) ön-tanımlı rollerinde ayrı analiz edilir ve hiçbir birincil
çıkarımsal analize karışmaz.
Geçiş-bandına kısıtlı model ve AR-only Friedman, ayrı estimand'lı ön-tanımlı
secondary analizlerdir. Hiçbir birincil sonuç, sonuca bağlı bir filtreden geçmez.

### 3. Betimsel — bozulma eğrileri (ana şekil)

GLMM'in karesel kestirimi eğrilerin üzerine bindirilir (uyum doğrulaması), σ₅₀
noktaları işaretlenir. Her (algoritma, indeks) için doğruluk × σ; satırlar ρ,
sütunlar gürültü yapısı (beyaz | AR .97). Gerçek verinin ölçülen bandı
(σ ≈ 0.35–0.81) taralı. **Nihai seçim AR(1) panelindeki taralı bant içinde en
yüksek doğruluğu veren (algoritma, indeks) çiftidir.** Beyaz panel literatür
karşılaştırması ve mekanizma tartışması içindir.

## Nihai seçim kuralı (winner rule — koşumdan önce dondurulmuştur)

**Havuz:** 4 algoritma × **4 birincil formülasyon** = **16 çift**. Exploratory
formülasyonlar (sil_cos, Dunn d2/D2 · d4/D1 · d3/D2) winner'a uygun değildir —
aile-temsilcisi mantığı test ve seçim katmanlarına simetrik uygulanır; aksi halde
"temsilciyi performansla seçmedik" savunması seçim katmanında çökerdi.

**Popülasyon:** AR(1) φ=0.97 hücreleri ∩ **nominal** σ ∈ {0.4, 0.5, 0.6, 0.7, 0.8},
her iki kol, üç ρ, beş k_true — 150 hücre, hepsi eşit ağırlıklı. (SSA'da hangi alt
rejimin geçerli olduğu bilinmiyor; eşit ağırlık en az varsayımlı seçim.)

**Nominal-σ kararı (donduruldu):** band nominal σ etiketiyle tanımlıdır,
`sigma_achieved` ile yeniden eşleştirilmez. Gerekçe: (i) ampirik band (0.35–0.81)
ile nominal etiket arasındaki AR-kaynaklı efektif-sapma küçülmesi (~%21 @ σ=0.7)
tam da ±0.1 band-kaydırma sağlamlık kontrolünün kapsadığı belirsizliktir;
(ii) QC tablosu nominal↔gerçekleşen eşlemesini raporladığı için okuyucu bandı
yeniden konumlandırabilir. Bu bir hakem sorusu olarak beklenir ve gerekçe metne
yazılmıştır.

**Skor:** her çift için 150 hücrenin aritmetik ortalama accuracy'si.

**Karar:** en yüksek skorlu çift kazanır — argmax, eşiksiz, sıra-kırıcısız,
deterministik.

**Raporlama yükümlülüğü (karar mekanizması değil):** birinci–ikinci farkı
**Monte Carlo belirsizliğiyle** raporlanır. 150 hücre rastgele örneklenmiş bir
popülasyon değil sabit tasarım noktalarıdır; SE bu yüzden tasarım-ötesi genelleme
belirsizliği değil, sonlu-tohum MC belirsizliğidir. Hesap: hücre c, tohum s için
eşleştirilmiş fark `d_cs = correct_winner − correct_runner` (aynı gerçekleşme);
hücre-içi MC varyansı d_cs'lerden, toplam SE 150 sabit hücre üzerinden birleştirilir.
Hücreler-arası performans heterojenliği ayrı bir niceliktir: hücre-başına ortalama
farkların SD/IQR/min-maks'ı ve faktör-katmanlı dağılımı betimsel raporlanır.
**Winner's-curse kabulü:** kazanan aynı veri üzerinde argmax ile seçildiğinden
birinci–ikinci farkının kendisi yukarı yanlıdır; fark küçükse dil "Y istatistiksel
olarak ayırt edilemez" DEĞİL, **"the estimated difference between X and Y was small
relative to its Monte Carlo uncertainty"** olur ve SSA uygulamasında Y ile de k
seçiminin değişip değişmediği ek raporlanır. Ortalama |bias| ve B11 accuracy'si
yakın yarışta okuyucunun tartacağı ikincil kanıt olarak tabloya girer — karar
zincirinde değildirler.

**Sağlamlık kontrolleri:** (i) σ bandı ±0.1 kaydırıldığında (→ {0.3–0.7} ve
{0.5–0.9}) birinci sıra değişiyor mu — raporlanır. (ii) **Achieved-σ transfer
duyarlılığı (ön-tanımlı):** AR(0.97) hücreleri içinden hücre-medyan
`sigma_achieved` (**transfer-uyumlu kestirimci — örneklem-prototip artık sapması,
§Gerçekleşen geometri QC**; `sigma_generator_deviation` DEĞİL) ∈ [0.35, 0.81]
(ampirik bandın kendisi) olan hücreler seçilir ve
winner skoru bu popülasyonda yeniden hesaplanır. Ölçüt yalnız manipülasyonun
gerçekleşen şiddetini kullanır, hiçbir CVI çıktısına bakmaz — outcome-tabanlı
değildir. Seçilen hücre sayısı ve faktör dağılımı raporlanır (popülasyon dengeli
olmayabilir; bu bir duyarlılıktır, karar mekanizması değil). Nominal-σ kararının
doğrudan dış-geçerlilik testidir ve ±0.1 kaydırma ile birlikte tutulur.
(iii) **Opsiyonel ek (karar-dışı, önceden adlandırıldı):** leave-one-
configuration-out winner kararlılığı — 30 yapısal konfigürasyondan biri sırayla
çıkarılıp birinci sıranın değişip değişmediğine bakılabilir; zorunlu değildir,
yapılırsa supplementary olarak raporlanır ve karar zincirine girmez.
(Configuration-düzeyi bir rastgele etki ise REDDEDİLDİ: 30 konfigürasyon
örneklenmiş küme değil, sabit ön-tanımlı faktöriyel tasarım noktalarıdır —
yorum-sınırı diliyle tutarlı.) v4'teki hücre-sayısı-ağırlığı kontrolü
**kaldırıldı**: winner popülasyonu her faktörde tam dengeli olduğundan (2×3×5×5,
doğrulandı) eşit ağırlık ile hücre-sayısı ağırlığı özdeş sonuç verir; kontrol bilgi
üretmez. Önerilen alternatif "ölçülen SSA geometrisine dayalı empirical weighting"
**benimsenmedi**: o geometri, gerçeği bilinmeyen bir kümelemeden ölçülür ve Blok
D'de kapatılan daireselliğin hafif biçimini karar zincirine geri sokar; en fazla
karar-dışı exploratory olarak düşünülebilir (şimdilik kapsam dışı).

**Friedman'ın rolü:** winner'ı seçmez; **önceden tanımlanmış birincil formülasyon
karşılaştırmasının** çıkarımsal kanıtını sağlar. Winner seçimi betimsel argmax
kuralına dayanır; havuz birincil setle özdeş olduğundan kazanan her durumda birincil
testin kapsamındadır (v4'teki 8-havuz/5-test uyumsuzluğu yapısal olarak kapandı).

**k aralığı notu:** aday aralığı 2–10, SSA uygulamasında da aynı aralıkla
**deployment-matched**. k_true=8 hücrelerinde `bias` asimetrik sınır etkisi taşır;
bias özetleri k_true'ya koşullu raporlanır, havuzlanmaz.

## SSA uygulaması (deployment) — dondurulmuş boru hattı

Kazanan gerçek veriye uygulanmadan önce tek ve seçeneksiz bir boru hattı
dondurulur:

```
SSA oran (ratio) yörüngeleri, 1880–2025
→ seri dahil-etme kuralı: kalibrasyonda kullanılan tam-seri dosyaları
  (top-N filtreli, 39 erkek / 57 kadın; eksik yılı olmayan seriler)
→ satır-bazlı z-normalizasyon (ddof=0)
→ kazanan algoritma, aday k = 2–10
→ kazanan CVI ile k̂ seçimi
```

**Aynılık kuralları:** sentetik ve gerçek tarafta aynı CVI implementasyonu
(registry kodu), aynı aday k aralığı ve aynı yön/bağ/non-finite politikası
(§k̂ seçimi) kullanılır — deployment'ta yeni hiçbir seçim yapılmaz.

**Birincil deployment kapsamı (onay bekliyor):** kalibrasyon dosyalarının
kendisi. Tam-SSA'ya genişleme, `sigma_hat`'in tam veride yeniden ölçülmesini
gerektirir (§Transfer) ve ayrı bir ön-kayıtlı genişletme adımıdır — bu
protokolün kapsamı dışındadır.

**Sınır-k uyarısı (önceden kayıtlı):** kazanan SSA'da k̂ = 2 veya k̂ = 10 seçerse
sınır uyarısı verilir; özellikle k̂ = 10, "optimum aday aralığının ötesinde
olabilir" anlamına gelir. Aralık deployment-matched seçildiğinden post-hoc
genişletilmez; sınır durumu olduğu gibi raporlanır. Yakın yarışta runner-up
çiftin SSA'daki k̂'si de raporlanır (§Winner raporlama yükümlülüğüyle tutarlı).

**Extrapolation sınırı (önceden kayıtlı):** tam SSA verisinde yeniden ölçülen
ampirik σ bandı sentetik kalibrasyon aralığının dışına çıkarsa sonuç dili
**"extrapolation-limited"** olur; winner bandını, simülasyon σ ızgarasını veya
winner popülasyonunu post-hoc yeniden tanımlamak yasaktır — gerekirse bu durum
bağımsız bir genişletme çalışmasının gerekçesi olur.

## Toplam maliyet

| Blok | Koşum (hücre×algoritma) | Tohum |
|---|---|---|
| A | 2400 | 100 |
| B | 60 (15 koşul satırı) | 100 |
| C | 48 | 100 |
| D | 16 | 100 |
| **Toplam** | **2524** | — |

Zaman serisi yolunda n ≤ 100; koşum başına maliyet düşük (saniyeler). Sürücü betiği
şart — `run_matrix_v4.csv` satırlarını okur, `TimeSeriesSyntheticDataGenerator` +
`optimal_k_analysis` çağırır, düz kayıt tablosu yazar.

## Koşum altyapısı ve yürütme bütünlüğü (donduruldu)

Metodoloji değil yürütme disiplini; final freeze'in parçasıdır.

**Hacim:** 631 hücre × 100 tohum × 9 aday k × 4 algoritma = **2,271,600 fit
prosedürü** (KMeans payı 567,900 fit × n_init 50 ≈ 28.4M Lloyd başlatması; GMM
567,900 × n_init 5 ≈ 2.84M EM koşusu). Koşum öncesi bir temsilci Blok A hücresi
ve bir temsilci Blok B hücresi (B11) benchmark'lanır; toplam süre, paralelleştirme
planı (worker sayısı, batch boyutu), checkpoint/restart politikası (kayıt
(hücre, algoritma, tohum) granülünde idempotent — tamamlanmış granüller yeniden
başlatmada atlanır) ve log/disk kapasitesi bu benchmark'a göre koşumdan önce
yazılır.

**Kritik kural:** sonuçlar görüldükten sonra runtime gerekçesiyle `n_init`,
`max_iter`, `tol`, aday k aralığı, tohum sayısı veya herhangi bir algoritma
parametresi DEĞİŞTİRİLEMEZ. Yalnız yürütme altyapısı (worker sayısı, batch,
checkpoint sıklığı, zamanlama) serbesttir.

**GMM izleme (secondary QC):** özellikle stres rejimlerinde — B1 (n_per=5),
yüksek k adayları, yüksek σ — `degenerate_candidate_rate`, GMM yakınsama
başarısızlık oranı ve `algorithm_failure` oranı raporlanır. `reg_covar=1e-6`
kararı yeniden açılmaz; bu izleme parametre ayarı için KULLANILMAZ, yalnız
raporlanır.

## Koşum-öncesi kontrol listesi

1. `run_matrix_v4.csv` üretildi ve doğrulandı: 631 satır; v3'ün 612 satırı
   bit-değişmez, `hucre_id` dondu; `seed_key_hucre` kurala uygun (yalnız B0 farklı).
2. Blok D prototip setleri (2 × `.npy`) üretildi, hash'ler kaydedildi.
3. ACF uyum kontrolü yapıldı, makale dili buna göre seçildi.
4. Registry singleton/dejenere davranış birim testleri geçti (silhouette s(i)=0,
   Dunn Δ_singleton=0, ortak geçersizlik kuralı, all-invalid → correct=0).
5. AR üretimi durağan-başlatmalı ve birim varyanslı — birim testi: Var(x_0) ≈ 1,
   ilk zaman noktalarının örneklem varyansı nominalde; ölçekli-innovation başlatma
   yok. Ön-ölçüm kodunun hangi başlatmayı kullandığı belgelendi, tarihsel not
   kesinleştirildi.
6. Seed mimarisi testi: aynı (seed_key, tohum) → özdeş X; farklı hücreler → ayrı
   akış; B0, R hücresini birebir yeniden üretiyor.
7. z-norm `ddof=0` birim testi: her z-normalize satır için ‖x‖² = T (tam).
8. σ₅₀, winner rule, agreement tanımları koda sabit parametre olarak girdi.
9. k̂ yön/bağ/non-finite politikası birim testleri: DB minimize; bağ kümesi
   en-iyi skora göre `isclose(rtol=1e-10, atol=1e-12)`; bağda min k +
   `k_hat_tie`; non-finite aday yalnız o CVI için düşer; `cvi_failure` ≠
   `algorithm_failure`.
10. İki QC kestirimcisi (`sigma_generator_deviation`, `sigma_achieved`) ve
    `rho_max_achieved`/`rho_max_pair` implementasyonu R hücresinde el hesabıyla
    doğrulandı.
11. Runtime benchmark (1 temsilci A + 1 temsilci B hücresi) koşuldu;
    paralelleştirme/checkpoint/disk planı yazıldı.
12. `blme` kurulumu ve Aşama-4 prior yapılandırması deneme verisiyle doğrulandı.
13. SSA deployment boru hattı (dahil-etme kuralı dahil) onaylandı ve dondu.

## Transfer varsayımı ve sınırlamalar

- Sonuçlar "sentetikte kazanan gerçekte de kazanır" varsayımına dayanır. İki
  kalibrasyon destekler: σ bandı (0.35–0.81, taralı) ve gürültü yapısı (lag-1
  AC ≈ 0.98 → seçim AR(1) panelinden). Beyaz-gürültü-yalnız tasarımda sıralamanın
  tersine döndüğü ölçülmüştür; AR(1)'in ana faktör olması pazarlık dışıdır.
- `sigma_hat` beş kümeleme yöntemiyle ölçüldü; yöntemler arası fark ≤ %17, k=8'de
  sıfır. Döngüsellik gerçek ama etkisi küçük.
- Kümeler arası yayılım farkı gerçek veride 2.1×'e kadar; ana blok eşit σ kullanır,
  fark Blok B'de test edilir.
- Gerçek dosyalar top-N filtreli alt kümeler (39 ve 57 isim). Tam SSA verisinde
  `sigma_hat` muhtemelen daha yüksek; tam veriye geçmeden önce ölçüm tekrarlanmalı.
- **Geometri kapsama kontrolü (koşum sonrası, makale öncesi):** gerçek SSA
  kümelemelerinden ölçülen `rho_max`, `rho_mean`, `d_eff` değerlerinin Blok A
  hücrelerinin kapladığı bölgeye düştüğü gösterilir; Blok D'nin ampirik prototip
  geometrisi de aynı tabloya girer. **Birincil kapsama kestirimcisi (donduruldu):**
  bu değerler winner'ın seçtiği kümelemeden DEĞİL, Blok D için dondurulmuş
  winner-bağımsız çapa olan **average-linkage k=5 prototiplerinden** ölçülür —
  kazananın kendi geometrisini doğrulaması döngüsü burada da kırılır. Duyarlılık
  olarak alternatif k değerleri ve birkaç alternatif kümeleme yöntemiyle kapsama
  sonucunun değişip değişmediği gösterilebilir. Kayıtlı akrabalık notu:
  average-linkage, Ward/Complete ile aynı geniş aglomeratif ailededir; yarışmacı
  değildir ve hiçbir yarışmacının amaç fonksiyonunu beslemez, ama aile yakınlığı
  limitations'ta tek cümleyle belirtilir.

---

## Onay durumu (v5.3 — dördüncü tur sonrası)

v5.1 onay listesi (8 madde) v5.2'de kapandı: B4 vektörü, counterbalance
hedefleri, mutlak σ_out=1.5, B11 hizalaması, Blok D average-linkage, GMM
sayısalları, PROTO_TAG onaylandı; all-invalid politikası `correct=0` +
`algorithm_failure=1` olarak değiştirildi.

Dördüncü turun 20 zorunlu maddesi v5.3'e işlendi. Karar raporunun Qwen'e iki
düzeltmesi aynen benimsendi: (i) transfer duyarlılığında `sigma_achieved` =
örneklem-prototip artık sapması (nominal-prototip uzaklığı değil); (ii)
zayıf-regülarize/Bayes model, 4-CVI frequentist yedeğinden SONRA gelir.
Configuration-düzeyi rastgele etki önerisi reddedildi (sabit faktöriyel tasarım
noktaları); LOCO kararlılığı opsiyonel-ek olarak önceden adlandırıldı.

**Freeze öncesi onay bekleyen üç karar (bu turda somutlaştırıldı):**

1. **`corr_ari`: TUT** (Spearman, yön-standardize, betimsel-yalnız) — önerilen
   karar bu; "çıkar" da savunulabilir, itiraz halinde tek alan silinir (§Kayıt).
2. **Aşama-4 paketi ve prior'ları:** `blme::bglmer`, sabit etkilere
   normal(0, 2.5²) (standardize prediktörlerde), kovaryansa paket-varsayılanı
   gamma prior (§GLMM yedek zinciri).
3. **SSA birincil deployment kapsamı:** kalibrasyon dosyalarının kendisi
   (39/57 tam seri); tam-SSA'ya genişleme ayrı ön-kayıtlı adım (§SSA).

Bu üç onay + kontrol listesinin icrasıyla protokol donar; yeni değerlendirme
turu açılmaz, sonraki her değişiklik "sapma" olarak kayda geçer.
