# SVM ile Meme Kanseri Teshisi (Breast Cancer Wisconsin Diagnostic)

Bu proje, **Support Vector Machine (SVM)** ile meme kanseri tani problemi uzerine iki farkli yaklasim sunar: kutuphanesiz (sifirdan) ve `scikit-learn` ile. Kaggle'daki **Breast Cancer Wisconsin (Diagnostic)** veri seti kullanilir.

## Veri Seti
- **Kaynak**: [Kaggle - Breast Cancer Wisconsin (Diagnostic) Data Set](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
- **Toplam ornek**: 569 hasta
- **Ozellikler**: 32 sutun (1 ID, 1 hedef etiket, 30 sayisal girdi)
- **Etiketler**: `M` = Malignant, `B` = Benign (preprocess adimi etiketleri {1, -1} olarak kodlar)

## Kullanım
1. Ham CSV'yi normalize et:
   ```bash
   python BreastCancer/preprocess.py --input BreastCancer/data.csv --output BreastCancer/processed_data.csv
   ```
2. Kutuphanesiz SVM (deterministik bolme ve egitim):
   ```bash
   python BreastCancer/svm_from_scratch.py --data-path BreastCancer/processed_data.csv --epochs 120 --lr 0.001 --C 10 --seed 42 --train-ratio 0.7 --val-ratio 0.15 --output-path BreastCancer/test_results_scratch.csv
   ```
3. scikit-learn SVM (plotlar opsiyonel):
   ```bash
   python BreastCancer/svm_with_sklearn.py --data-path BreastCancer/processed_data.csv --kernel linear --C 10 --gamma scale --seed 42 --skip-plots --output-path BreastCancer/test_results_sklearn.csv
   ```

Notlar:
- Tum scriptler varsayilan olarak `BreastCancer/processed_data.csv` yolunu kullanir ve `--seed` ile ayni bolumler yeniden uretilebilir.
- `--skip-export` tahmin CSV'sini yazmaz; `--skip-plots` / `--show-plots` bayraklari gorsel ciktilari kontrol eder.

## Script Ozeti

### 1) Kutuphanesiz SVM (`BreastCancer/svm_from_scratch.py`)
- Lineer SVM icin hinge loss ve L2 regularization uygular.
- `data_utils.split_data` ile deterministik train/val/test bolumu.
- Test tahminleri opsiyonel olarak `test_results_scratch.csv` dosyasina yazilir.

### 2) scikit-learn SVM (`BreastCancer/svm_with_sklearn.py`)
- `StandardScaler` + `sklearn.svm.SVC` (varsayilan kernel: linear).
- Test tahminleri CSV'ye yazilabilir, karisiklik matrisi ve 2B PCA karar siniri grafiklerini kaydedebilir.
- Plotlar `--skip-plots` ile devre disi birakilabilir; gorseller `BreastCancer/images/` dizinine kaydedilir.

## Ornek Ciktilar
- Kutuphanesiz SVM egitim logu ve performans: `BreastCancer/images/svm_scratch_results.jpeg`
- scikit-learn modeli performansi: `BreastCancer/images/svm_sklearn_results.jpeg`
- Test karisiklik matrisi: `BreastCancer/images/confusion_matrix.jpeg`
- PCA uzerinde karar siniri: `BreastCancer/images/decision_boundary_pca.png`
