import pytest
import numpy as np
from LogisticRegression import train_logistic_regression, accuracy, predict, sigmoid

@pytest.fixture
def trained_model():
    """
    Eğitim yapılmış model ve test verilerini dönen fixture.
    """
    final_w, final_b, X_test, y_test = train_logistic_regression()
    return final_w, final_b, X_test, y_test

def test_accuracy(trained_model):
    """
    Model doğruluğunu kontrol eder.
    """
    final_w, final_b, X_test, y_test = trained_model
    y_pred = predict(X_test, final_w, final_b)
    acc = accuracy(y_test, y_pred)
    print(f"Model doğruluğu: {acc:.4f}")
    assert acc >= 0.7, f"Test accuracy düşük kaldı: {acc}"

def test_prediction_shape(trained_model):
    """
    Tahmin edilen çıktının boyutunu kontrol eder.
    """
    final_w, final_b, X_test, y_test = trained_model
    y_pred = predict(X_test, final_w, final_b)
    assert y_pred.shape == y_test.shape, "Tahmin boyutu y_test boyutuyla eşleşmiyor"

def test_no_nan_predictions(trained_model):
    """
    Tahminlerin NaN içermediğini kontrol eder.
    """
    final_w, final_b, X_test, _ = trained_model
    y_pred = predict(X_test, final_w, final_b)
    assert not np.isnan(y_pred).any(), "Tahminler NaN değerler içeriyor"

def test_prediction_binary(trained_model):
    """
    Tahminlerin sadece 0 ve 1 değerlerini içerdiğini kontrol eder.
    """
    final_w, final_b, X_test, _ = trained_model
    y_pred = predict(X_test, final_w, final_b)
    assert np.all(np.logical_or(y_pred == 0, y_pred == 1)), "Tahminler 0 ve 1 dışında değerler içeriyor"

def test_sigmoid_bounds():
    """
    Sigmoid fonksiyonunun 0-1 arasında değerler ürettiğini kontrol eder.
    """
    test_values = np.array([-100, -10, 0, 10, 100])
    sigmoid_values = sigmoid(test_values)
    assert np.all(sigmoid_values >= 0) and np.all(sigmoid_values <= 1), "Sigmoid değerleri 0-1 aralığında değil"

def test_weights_not_zero(trained_model):
    """
    Ağırlıkların sıfır olmadığını kontrol eder.
    """
    final_w, final_b, _, _ = trained_model
    assert not np.all(final_w == 0), "Tüm ağırlıklar sıfır"
    assert final_b != 0, "Bias terimi sıfır"

def test_prediction_range(trained_model):
    """
    Tahmin olasılıklarının 0-1 arasında olduğunu kontrol eder.
    """
    final_w, final_b, X_test, _ = trained_model
    z = np.dot(X_test, final_w) + final_b
    probabilities = sigmoid(z)
    assert np.all(probabilities >= 0) and np.all(probabilities <= 1), "Tahmin olasılıkları 0-1 aralığında değil"

def test_model_consistency(trained_model):
    """
    Aynı giriş için tutarlı tahminler yapıldığını kontrol eder.
    """
    final_w, final_b, X_test, _ = trained_model
    predictions1 = predict(X_test, final_w, final_b)
    predictions2 = predict(X_test, final_w, final_b)
    assert np.array_equal(predictions1, predictions2), "Model tutarsız tahminler yapıyor"