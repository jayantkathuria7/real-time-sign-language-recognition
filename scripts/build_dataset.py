from src.data.dataset_builder import build_and_save_data
import time

def main():
    start_time=time.perf_counter()
    X_shape, y_shape =build_and_save_data()
    print("Dataset built successfully! --> took", time.perf_counter()-start_time)
    print("X shape:", X_shape)
    print("y shape:", y_shape)

if __name__ == "__main__":
    main()