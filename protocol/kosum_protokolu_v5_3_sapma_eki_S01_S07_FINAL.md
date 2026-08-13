# Koşum Protokolü v5.3 — Sapma Eki S-01…S-07

**STATÜ: FINAL / APPROVED / FROZEN**
**Onay tarihi:** 2026-08-12
**Onay artefaktı:** `claude_nihai_onay_promptu.md` (nihai kullanıcı onayı;
Onay Bloğu v4 KAPANDI)
**Değerlendirme kaydı:** `final_denetim_karar_kaydi_S01_S07.md` (rev-0…rev-3) —
tarihsel kayıt; protokol düzeyinde bağlayıcı metin BU ektir.
**Yürürlük:** kanonik `kosum_protokolu_v5_3.md` + bu ek = yürürlükteki
protokol. Bundan sonraki her değişiklik yeni, tarihli sapma kaydıdır.
Metodolojik değerlendirme döngüsü kapanmıştır; yeni öneri/eşik/model/optimizer
turu açılmaz.

**Onayda yapılan ifade temizliği:** rev-3'teki "near-separation … hiçbir
separation dedektörünün hedefinde değildir" ikincil cümlesi kullanıcı
talimatıyla ÇIKARILMIŞTIR; bağımsız separation tanısı kullanmama kararının
gerekçesi yalnız aşağıdaki kapsam argümanıdır. Metodolojik içerik değişikliği
değildir.

---

## v5.3 metnini değiştiren sapmalar (tam liste — 5 nokta)

1. **Winner bölümü:** "sıra-kırıcısız" cümlesine S-01'in co-winner /
   öncelik-sırası / koşulsuz runner-up / SE_MC paragrafı eklenir.
2. **GLMM Aşama-2 cümlesi:** "alternatif optimizer (bobyqa; ikinci alternatif
   Nelder-Mead)" → S-02'deki tek yeniden-koşum metni
   (`optimizer = c("bobyqa", "bobyqa")`); Nelder-Mead alternatifi silinmiştir.
3. **GLMM Aşama-4 cümlesi:** "kovaryansa paket-varsayılanı **gamma** prior"
   (olgusal hata) → çözümlenmiş **wishart**; "(standardize prediktörlerde)" →
   "(σ_c = σ − 0.55 merkezli ölçekte; ek ölçekleme yok)".
4. **Manifest:** "boş = referans değer" → "boş = yalnız NA/not-applicable";
   "Blok-D geometri alanları QC sonrası doldurulur" → "prototip-dondurma
   anında ön-yazım".
5. **Yedek zinciri tetik listesi:** "separation" kelimesinin bağımsız
   operasyonalizasyonu yoktur; convergence + singularity tetiklerine
   massedilmiştir.

Diğer tüm maddeler ekleyici pindir; hiçbir kapalı tasarım kararı
değişmemiştir.

---

## S-01 — Winner: exact tie, co-winner, koşulsuz runner-up, SE_MC

Winner skoru tam oran `total_correct / 15000` olarak hesaplanır; karşılaştırma
**tamsayı** `total_correct` üzerinden yapılır (exact tie kayan-nokta
toleranssız saptanır). Exact eşitlikte bağlı çiftler **co-winner** olarak
kaydedilir — bilimsel iddia düzeyinde birincil kayıt budur ve "sıra-kırıcısız"
ilkesi korunur. Tek deployment yolu gerektiğinde, yalnız teknik amaçla ve
sonuçtan bağımsız sabit öncelik sırası kullanılır: **16 çift üzerinde
algoritma-majör leksikografik sıra**; algoritma sırası (KMeans, Ward,
Complete, GMM), CVI sırası (sil_euc, DB, CH, Dunn d1/D1):
1 (KMeans, sil_euc), 2 (KMeans, DB), 3 (KMeans, CH), 4 (KMeans, d1/D1),
5 (Ward, sil_euc), 6 (Ward, DB), 7 (Ward, CH), 8 (Ward, d1/D1),
9 (Complete, sil_euc), 10 (Complete, DB), 11 (Complete, CH),
12 (Complete, d1/D1), 13 (GMM, sil_euc), 14 (GMM, DB), 15 (GMM, CH),
16 (GMM, d1/D1).

**Runner-up koşulsuz raporlanır:** çift, skor, Δaccuracy, SE_MC, hücre-bazlı
paired farkların SD/IQR/min–maks'ı ve SSA k̂'si her durumda verilir.
Co-winner varlığında runner-up = kazanan skorun **kesinlikle altındaki** en
yüksek `total_correct` değerine sahip çift (co-winner kümesi dışından);
runner-up düzeyinde de exact tie varsa kümenin tamamı raporlanır.

**SE_MC = sqrt(Σ_c s_c² / S) / C**, C = 150, S = 100; `s_c²` tohumlar
üzerinde **ddof = 1**. SE_MC yalnız Monte Carlo belirsizliğidir; hipotez veya
eşdeğerlik testi değildir. Eklenmeyen eşikler (reddedilmiş kalır):
`1.96×SE_MC`, `|Δ|<0.02`, `|Δ|<0.05+IQR/2`.

## S-02 — GLMM: yedek zinciri, tetikler, exact Aşama-4, prediktör ölçeği

**Yedek zinciri (Seçenek B; 4 aşama; algoritma başına):**

- **Aşama 1:** 8-CVI frequentist `glmer`;
  `control = glmerControl(optimizer = c("bobyqa", "Nelder_Mead"))` — açık
  yazım (iki-fazlı yapı: nAGQ0 → bobyqa, final → Nelder_Mead; lme4
  varsayılanları tarihsel olarak değiştiğinden açık yazım drift korumasıdır).
  Diğer control alanları pinli sürümün varsayılanıdır ve `sessionInfo()` ile
  kayda geçer.
- **Tetik seti (her aşama geçişinde aynı):** {herhangi bir lme4 convergence
  uyarısı} ∪ {`isSingular`, tol = pinli sürümün çözümlenmiş varsayılanı
  (kaynak: 1e-4; `lme4.singular.tolerance` opsiyonu koşum ortamında
  değiştirilmez)}. **Bağımsız separation tanısı/tetikleyicisi yoktur** —
  pre-fit LP veya post-fit SE-oran dahil. Gerekçe: `detect_separation`
  binomial GLM tasarım matrisi için tanımlıdır; model random-intercept'li bir
  GLMM'dir; sabit-etki tasarımı üzerinde ayrı bir GLM separation testi
  mixed-model likelihood patolojisiyle bire bir eşdeğer değildir; GLMM bu
  çalışmada confirmatory test değil effect-estimation katmanıdır; terminal
  `bglmer` regularization sağlar. Kayıtlı sınırlama: temiz-yakınsamış ama
  patolojik bir fit aşama tetiklemez — kabul edilmiştir.
- **Aşama 2:** aynı 8-CVI model, **tek** yeniden koşum;
  `control = glmerControl(optimizer = c("bobyqa", "bobyqa"))` — final-faz
  optimizerini değiştirir; yaygın ve belgelenmiş bir çözüm yolu olup burada
  önceden kayıtlı fallback tercihi olarak dondurulmuştur (tekil "kanonik yol"
  iddiası yoktur).
- **Aşama 3:** 4-primary-CVI frequentist model; Aşama-1 control'ü.
- **Aşama 4:** exact `bglmer`:

```r
bglmer(
    correct ~ CVI + sigma_c + I(sigma_c^2) + rho_f + k_f + gurultu + kol
              + CVI:sigma_c + CVI:I(sigma_c^2)
              + CVI:gurultu + CVI:kol
              + (1 | hucre) + (1 | hucre:tohum),
    family = binomial,
    control = glmerControl(optimizer = c("bobyqa", "Nelder_Mead")),
    fixef.prior = normal(sd = 2.5),
    cov.prior = wishart(df = level.dim + 2.5, scale = Inf,
                        posterior.scale = "cov")
)
```

**Prior spesifikasyonu:** kovaryans = **W / Wishart** (paketin gerçek
varsayılanıyla uyumlu; v5.3'teki "paket-varsayılanı gamma" ifadesi olgusal
hataydı ve düzeltilmiştir); iki skaler random-intercept bloğunda çözümlenmiş
hali `wishart(df = 3.5, scale = Inf, posterior.scale = "cov")` (`level.dim`
blme'nin prior-değerlendirme ortamında çözülür). Sabit etkiler:
**Normal(0, 2.5²), kesişim dahil tüm katsayılar**; paket ikilisi `c(10, 2.5)`
kullanılmaz.

**Prediktör ölçeği (reproducibility pini):** `sigma_c = sigma − 0.55`
(0.55 = σ ızgarasının tasarım ortalaması); **başka hiçbir
scaling/standardization uygulanmaz** — SD'ye bölme yok, yeniden ölçekleme
yok; `I(sigma_c^2)` merkezlenmiş değerin karesidir; σ tek sürekli
prediktördür; tüm faktörler (CVI, rho_f, k_f, gurultu, kol) treatment kodludur
ve ölçeklenmez; referans seviyeler protokol listeleme sırasının ilk seviyesi:
**sil_euc · beyaz · P_konum · r045 · k=3**. Normal(0, 2.5²) prior'ı bu exact
ölçek altında tanımlıdır.

**σ₅₀ yorum sınırı:** çapraz-algoritma σ₅₀ kıyası yalnız ortak 600-hücre
full-data birincil Blok-A GLMM tahminlerinden yapılır; transition-kısıtlı
σ₅₀/eğim/olasılık kestirimleri yalnız algoritma içinde yorumlanır.
Eklenmeyen (reddedilmiş kalır): KMeans+sil_euc referanslı ortak transition
bandı.

**Checklist-12 teyitleri (yürütme kontrolü):** `sessionInfo()`; pinli
sürümde `formals(glmerControl)`, `formals(bglmer)` ve prior kurucularının
karşılaştırılması; açık çağrı sözdiziminin kabulü; `normal(sd = 2.5)`
skaler-yayılım davranışı (yaymazsa niyeti açıkça sağlayan biçime çevrilir ve
kaydedilir); `level.dim` çözümlemesi. Fark çıkarsa açık çağrıdaki **değerler**
bağlayıcıdır; fark sapma olarak kaydedilir.

## S-03 — Manifest immutability

**(a) Boş hücre yalnız NA/not-applicable'dır; asla kod-varsayılanı değildir.**
Alan sınıfları:

- **Sınıf (i) — zorunlu üretim parametresi (koşulacak her satırda AÇIK, asla
  NA):** `sigma`, `gurultu`, `siniflar`, `n_per_cluster`, `jitter` (0|1),
  `aykiri_n` (0|3); AR satırlarında `phi`; heterojenliğin tanımlı olduğu
  satırlarda `kume_boyutlari` / `sigma_vektoru` (**vektör bağlayıcıdır**;
  o satırlarda `n_per_cluster` ve `sigma` nominal/türetilmiş etikettir).
- **Sınıf (ii) — mekanizma/kapsam alanı (kapsam dışında NA):**
  `aykiri_sigma` (yalnız `aykiri_n > 0` iken tanımlı), `atama_id` (yalnız
  counterbalance koşulları), `cinsiyet` ve `prototip_*` (yalnız D), `kol`
  (yalnız A), beyaz-gürültü satırlarında `phi`.
- **Sınıf (iii) — türetilmiş geometri:** A/C v3'ten; B = R (hucre 449)
  kopyası; D = prototip-dondurma anında ön-yazım.

**Blok C `n_per_cluster` NA olamaz:** v3'te sütun varsa bit-değişmez taşınır;
yoksa üretici koddan/tasarım kaydından **kullanıcı teyidiyle** yazılır —
tahmin yoktur; teyit gelmeden hash-freeze yapılamaz. **Sürücü fail-fast:**
koşulacak bir satırda sınıf-(i) alanı boşsa üretim durur.

**(b) Ön-yazım:** önceden hesaplanabilir D geometri alanları (`rho_max`,
`rho_mean`, `d_eff`, `theta_min`; σ verildiğinden `ratio`, `verdict`)
prototipler donduğu anda, koşumdan ÖNCE manifeste yazılır. Gerçekleşme
QC'leri (`sigma_achieved`, `rho_max_achieved`) yalnız sonuç tablosuna gider;
manifeste asla geri yazılmaz.

**(c) İki aşamalı finalizasyon:** v4-taslak (v3 gelince) → D prototipleri →
D alanları + `prototip_set_id` + `prototip_hash` yazımı → dosya **SHA256 ile
dondurulur** → koşum. Hash sonrası in-place değişiklik yasaktır; her
değişiklik = yeni dosya + sapma kaydı.

## S-04 — Failure/exception politikası ve GMM convergence loglama

Candidate-k fit exception → ilgili (hücre, tohum, algoritma, k) adayı
geçersizdir; exception loglanır; retry yoktur. CVI exception → yalnız ilgili
(CVI, k) adayı düşer; finite aday kalmazsa mevcut `cvi_failure = 1`. GMM
convergence **(hücre, tohum, k)** düzeyinde saklanır; convergence nedeniyle
hiçbir primary/winner dışlaması yapılmaz. Winner GMM ise winner-popülasyon
convergence/degeneracy/failure QC'si ana sonucun yanında raporlanır. Kayıt
şemasına iki alan eklenir: exception log + aday-k bazlı `converged`.
Eklenmeyen (reddedilmiş kalır): `<0.80`, `>%10`, converged-only alt-küme
analizleri.

## S-05 — `sigma_achieved` / `rho_max_achieved` formül pinleri

`P̂_c = z(mean_{i∈c} z_i)`;
`sigma_achieved = sqrt{ (1/(NT)) Σ_i Σ_t [z_it − P̂_{c(i),t}]² }` — pooled,
gözlem-ağırlıklı, **ddof=0**. (z-normalize seri − z-normalize prototip artığı
seri başına tam sıfır ortalamalı olduğundan merkezli SD ile merkezsiz RMS
burada özdeştir; v5.3'ün "SD_ddof0" ifadesiyle çelişki yoktur — pinlenen,
havuzlamadır.) SSA tarafında aynı kestirimci kestirilmiş etiketlerle
kullanılır. `rho_max_achieved = max_{c<d} (P̂_c^T P̂_d / T)` — **signed**
Pearson maksimum; `abs()` yasaktır; `rho_max_pair` aynı argmax çifttir.

## S-06 — Onay dili ve girdi provenance'ı

"Onay bekliyor" dili kaldırılmıştır — üç karar 2026-08-12'de onaylanmıştır:
`corr_ari = TUT` (Spearman, yön-standardize, betimsel-yalnız; <3
geçerli-finite aday → NA); Aşama-4 exact spesifikasyon (S-02'deki haliyle);
birincil SSA deployment = **39 erkek / 57 kadın kalibrasyon dosyaları**
(tam-SSA ayrı ön-kayıtlı genişletme). Kalibrasyon girdi dosyalarının
**filename + SHA256** değerleri koşumdan önce kaydedilir. Aynı CVI kod yolu,
k = 2–10, bağ/non-finite politikası, ddof=0, sınır-k ve extrapolation-limited
politikaları değişmeden korunur. Opsiyonel analizler yalnız "post-hoc
exploratory; winner/transfer kararını değiştirmez" etiketiyle raporlanır.

## S-07 — Friedman/Nemenyi multiplicity kapsamı

Protokole eklenen cümle: "Nemenyi post-hoc FWER kontrolü algoritma-içi CVI
karşılaştırma ailesi için tanımlıdır; dört algoritmanın tüm pairwise post-hoc
sonuçlarını kapsayan tek bir global cross-algorithm FWER = .05 iddiası
yapılmaz." Aynı sınır AR-only secondary aile için geçerlidir.

---

## Reddedilmiş kalanlar (yeniden açılmaz)

Yakın-yarış eşikleri (`1.96×SE_MC`, `|Δ|<0.02`, `|Δ|<0.05+IQR/2`); GMM
`<0.80` / `>%10` / converged-only kuralları; KMeans+sil_euc referanslı ortak
transition bandı; sayısal separation cutoff (`|β̂|>10` — silindi); bağımsız
separation dedektör bağımlılığı (geri çekildi).

---

## Yürütme/provenance ön-koşulları

Aşağıdakiler **metodolojik onay bekleyen kararlar DEĞİLDİR**; yalnız
execution/provenance kontrolleridir:

| # | Kalem | Taraf / durum |
|---|---|---|
| 1 | `run_matrix_v3.csv` yüklemesi | kullanıcı — 2 ve 9 bunu bekler |
| 2 | `run_matrix_v4.csv` üretimi + doğrulaması (iki aşamalı finalizasyon, S-03c) | Claude — betik hazır, v3 gelince koşar |
| 3 | Blok-D prototip dosyaları (2 × `.npy`) + SHA256 → D ön-yazımı + manifest hash-freeze | kullanıcı → Claude |
| 4 | SSA input filename + SHA256 | kullanıcı |
| 5 | Runtime benchmark (1 A + 1 B hücresi) | kullanıcı |
| 6 | ACF uyum + ön-ölçüm AR başlatma kodu kontrolleri | kullanıcı |
| 7 | `sessionInfo()` + pinli R/lme4/blme sürümleri | kullanıcı |
| 8 | Checklist-12 `formals()` / prior davranış doğrulaması (S-02 listesi) | kullanıcı |
| 9 | `proto_dogrulama` PASS (kontrol listesi madde 14) | Claude — v3 gelince koşar |
| 10 | Blok C `n_per` kaynak teyidi (yalnız v3'te sütun yoksa) | kullanıcı (koşullu) |

---

**S-01…S-07 REV-3 ONAYLANDI VE DONDURULDU. METODOLOJİK DEĞERLENDİRME DÖNGÜSÜ
KAPANDI. BUNDAN SONRA YALNIZ YÜRÜTME/PROVENANCE ÖN-KOŞULLARI
TAMAMLANACAKTIR.**
