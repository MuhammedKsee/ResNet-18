# 🔎 ResNet18 from Scratch (NumPy Only)

Bu proje, derin öğrenme dünyasının temel taşlarından biri olan **ResNet-18 mimarisinin** sıfırdan, yalnızca NumPy kullanılarak inşa edildiği bir Python uygulamasıdır. Hiçbir derin öğrenme framework'ü (PyTorch, TensorFlow) kullanılmamıştır.

---

## 📚 İçerik

Bu projede aşağıdaki yapılar sıfırdan yazılmıştır:

- ✅ Residual bloklar (`Conv2_x`, `Conv3_x`, `Conv4_x`, `Conv5_x`)
- ✅ ReLU aktivasyon fonksiyonu
- ✅ Residual (shortcut) bağlantılar
- ✅ Global Average Pooling
- ✅ Fully Connected (FC) katman
- ✅ Softmax fonksiyonu (binary sınıflandırma için)
- ✅ `predict()` fonksiyonu ile tam akış

---

## 🚀 Nasıl Çalışır?

### 1. Giriş

```python
x = np.array([1.0, -0.5, 2.0])
```

### 2. Ağırlıklar ve Biaslar

Rastgele 4 blok için, her biri 5 Conv katmanı olacak şekilde ağırlıklar ve biaslar üretilir:

```python
weights = [[np.random.randn(3) for _ in range(5)] for _ in range(4)]
biases  = [[np.random.randn() for _ in range(5)] for _ in range(4)]
```

### 3. Model Çalıştırma

```python
model = ResNet18()
output = model.predict(x, weights, biases)
```

### 4. Global Average Pooling

```python
gap_output = model.gap(output)
```

### 5. FC + Softmax

```python
W_fc = np.random.randn()
b_fc = np.random.randn()
fc_output = W_fc * gap_output + b_fc
softmax_output = 1 / (1 + np.exp(-fc_output))
```

---

## 📈 Örnek Çıktı

```
Sonuc: [1.74, 1.92, 2.49]
GAP Çıktısı: 2.05
FC Output: 0.95
Softmax: 0.72
```

---

## 🎯 Neden Bu Proje?

Bu proje, derin öğrenme mimarilerini:

- Derinlemesine anlamak
- Katman katman mantığını kavramak
- Otomatik kütüphaneler olmadan sıfırdan kurmak

için geliştirilmiştir. Eğitim veya ileri düzey optimizasyon şu an dahil değildir, ancak kolayca entegre edilebilir.

---

## ✍️ Geliştiren

📌 [Muhammed KSE](#)  
💡 "Mimariyi anlamadan model kullanmak, binanın temelini bilmeden kat çıkmaktır."

---

## 📌 Notlar

- NumPy dışındaki hiçbir kütüphane kullanılmamıştır.
- Tüm işlemler vektör düzeyindedir (Conv katmanlar basitleştirilmiştir).
- Dilersen `Global Average Pooling`, `Softmax`, `Cross Entropy Loss`, `Backpropagation` gibi eklentilerle geliştirmeye açıktır.

---

## 🧠 Katkı ve Devam

Eğer bu projeyi beğendiysen:

- ⭐ Star ver
- 🍴 Forkla
- 🤝 Pull Request gönder
```