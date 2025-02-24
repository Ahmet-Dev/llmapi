# llmapi

LLM API System bir Flask tabanlı web API uygulamasıdır. Bu API, Transformers kütüphanesinden AutoModelForCausalLM ve AutoTokenizer kullanarak büyük dil modelleriyle (örneğin Llama) çalışmayı amaçlıyor.

Kodun İşlevleri:

Bağımlılıkları Kurma
install_requirements() fonksiyonu, scikit-build, setuptools, flask, waitress, transformers, torch ve requests gibi gerekli Python paketlerini yüklüyor.

Model ve Tokenizer Yükleme

load_model_and_tokenizer(model_dir) fonksiyonu, belirli bir dizinden dil modeli ve tokenizer yüklemeye çalışıyor.
Eğer model dosyası (model.safetensors) yoksa, modeli indirme mekanizması da içerebilir.

HTTP Sunucu Uygulaması (Flask API)
Kullanıcı isteklerini almak ve işlemek için Flask kullanıyor.
Flask uygulaması, waitress ile sunucu olarak çalıştırılıyor.

İstek Günlüğü ve Kullanıcı Yönetimi
API, kullanıcıların kimliğini user_ids ile takip ediyor.
request_log değişkeni, yapılan API çağrılarını takip ediyor.
invalid_attempts ve blocked_ips değişkenleri, yetkisiz veya başarısız girişimleri yönetiyor.
