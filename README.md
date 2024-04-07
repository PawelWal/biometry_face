## W celu uruchomienia ewaluacji należy wykonać poniższe kroki:

1. Przygotować zbiór danych z folderami:
   - `train` - zbiór treningowy
   - `test_known` - zbiór testowy (znane klasy)
   - `test_unkown` - zbiór testowy (nieznane klasy)
     każdy z folderów powinien zawierać podfoldery z danymi dla poszczególnych klas, w formacie
     `{klasa}/{numer obrazu}.jpg` np. `train/1/image1.jpg`
2. Uruchomić skrypt `verify_app.py` z odpowiednimi parametrami:

```bash
python verify_app.py --data_dir={ścieka do folderu z danymi}
```

## W celu uruchomienia aplikacji należy wykonać poniższe kroki:

1. Przygotować zbiór danych z folderami:
   - `train` - zbiór treningowy
     folder powinien zawierać podfoldery z danymi dla poszczególnych klas, w formacie
     `{klasa}/{numer obrazu}.jpg` np. `train/1/image1.jpg`
2. podmienić ścieżkę do folderu z danymi w pliku `app_main.py`
3. Uruchomić aplikację poprzez uvicorn:

```bash
python -m uvicorn app_main:app --host 0.0.0.0 --port 8080`
```

Aplikacja będzie dostępna pod adresem `http://localhost:8080/docs`
