import numpy as np
import tensorflow as tf
from multiprocessing import Pool, Manager

def fully_flatten(arr, counter):
    stack = [arr]
    out = []
    
    while stack:
        current = stack.pop()
        if isinstance(current, (np.ndarray, list)):
            stack.extend(current)
        else:
            out.append(current)
    
    counter[0] += 1
    print(f"Processed {counter[0]} items")
    
    return np.array(out)

def process_flattened_data(args):
    i, counter = args
    return fully_flatten(i[9], counter)

def compute_cosine_similarity(tensor1, tensor2):
    normalized_tensor1 = tf.nn.l2_normalize(tensor1, axis=1)
    normalized_tensor2 = tf.nn.l2_normalize(tensor2, axis=1)
    return tf.reduce_sum(tf.multiply(normalized_tensor1, normalized_tensor2), axis=1)

def compute_similarities(round, counter):
    f1 = f"/data/jcl/big_batch_520/data_gradients_{round-1}_1.npy"

    data = np.load(f1, allow_pickle=True)
    g0 = np.load(f"/data/jcl/big_batch_520/model_gradients_{round}_0.npy", allow_pickle=True)
    g1 = np.load(f"/data/jcl/big_batch_520/model_gradients_{round}_1.npy", allow_pickle=True)

    with Pool() as pool:
        flattened_data = pool.map(process_flattened_data, [(i, counter) for i in data])

    with tf.device('/GPU:0'):
        flattened_data_tensor = tf.convert_to_tensor(flattened_data, dtype=tf.float32)

        flattened_g0 = fully_flatten(g0[9], counter)
        flattened_g0_tensor = tf.convert_to_tensor(flattened_g0.reshape(1, -1), dtype=tf.float32)
        similarities0 = compute_cosine_similarity(flattened_data_tensor, flattened_g0_tensor).numpy().flatten()
        np.save(f"similarities_{round}_0.npy", similarities0)

        flattened_g1 = fully_flatten(g1[9], counter)
        flattened_g1_tensor = tf.convert_to_tensor(flattened_g1.reshape(1, -1), dtype=tf.float32)
        similarities1 = compute_cosine_similarity(flattened_data_tensor, flattened_g1_tensor).numpy().flatten()
        np.save(f"similarities_{round}_1.npy", similarities1)

# 处理 round 从 550 到 600 的所有文件
if __name__ == "__main__":
    manager = Manager()
    counter = manager.list([0])
    
    rounds = range(451, 500)
    for round in rounds:
        compute_similarities(round, counter)
