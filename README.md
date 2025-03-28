# ResNet-18 Basitleştirilmiş Implementasyonu

Bu proje, ResNet-18 mimarisinin basitleştirilmiş bir implementasyonunu içermektedir. Proje, yapay sinir ağları ve konvolüsyonel katmanların temel prensiplerini göstermektedir.

## Proje Yapısı

Proje iki ana Python dosyasından oluşmaktadır:

- `ArtificalNeuralNetwork.py`: Temel yapay sinir ağı bileşenlerini içerir
- `resnet-18.py`: ResNet-18 mimarisinin basitleştirilmiş implementasyonunu içerir

## Gereksinimler

- Python 3.x
- NumPy

## Kurulum

1. Projeyi klonlayın
2. Gerekli kütüphaneleri yükleyin:
```bash
pip install numpy
```

## Kullanım

Projeyi çalıştırmak için:

```python
from resnet-18 import ResNet18

# Model oluşturma
model = ResNet18()

# Örnek girdi ve ağırlıklar
x = np.array([1.0, -0.5, 2.0])
w1 = np.random.randn(3)
b1 = np.random.randn()
w2 = np.random.randn(3)
b2 = np.random.randn()

# Tahmin
output = model.predict(x, w1, b1, w2, b2)
print("Sonuç:", output)
```

## Özellikler

- ReLU aktivasyon fonksiyonu
- Konvolüsyonel katmanlar
- Skip connection (atlama bağlantısı) implementasyonu
- Basitleştirilmiş ResNet-18 mimarisi

## Not

Bu implementasyon, orijinal ResNet-18 mimarisinin basitleştirilmiş bir versiyonudur ve eğitim amaçlıdır. Gerçek uygulamalar için PyTorch veya TensorFlow gibi derin öğrenme framework'lerinin kullanılması önerilir. 