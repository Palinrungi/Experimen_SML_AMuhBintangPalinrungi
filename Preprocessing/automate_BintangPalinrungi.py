# ...  (kode sebelumnya tetap sama) ... 

def main():
    """
    Fungsi main untuk menjalankan preprocessing
    """
    # Initialize preprocessor
    preprocessor = CreditCardFraudPreprocessor()
    
    # ✅ FIXED: Path ke dataset yang benar
    # Cek beberapa kemungkinan path
    possible_paths = [
        '../Credit Card Dataset_raw/credit_card_fraud_10k.csv',  # Dari Preprocessing/ folder
        'Credit Card Dataset_raw/credit_card_fraud_10k.csv',     # Dari root
    ]
    
    data_path = None
    for path in possible_paths:
        if os.path.exists(path):
            data_path = path
            break
    
    if data_path is None:
        print("❌ Dataset tidak ditemukan!")
        print("Mencari di:")
        for path in possible_paths:
            print(f"  - {path}")
        return
    
    print(f"✅ Dataset ditemukan: {data_path}")
    
    # ✅ FIXED: Output directory yang benar
    output_dir = 'CreditCard_Preprocessing'  # Output di Preprocessing/CreditCard_Preprocessing
    
    # Jalankan preprocessing
    processed_data = preprocessor.preprocess(
        filepath=data_path,
        test_size=0.2,
        random_state=42,
        balance=True,
        save_preprocessor=True,
        output_dir=output_dir  # ✅ Updated
    )
    
    print("\n✅ Preprocessing selesai!  Data siap untuk training.")
    print(f"📊 X_train shape: {processed_data['X_train'].shape}")
    print(f"📊 X_test shape: {processed_data['X_test'].shape}")
    print(f"📊 Number of features: {len(processed_data['feature_columns'])}")


if __name__ == "__main__":
    main()
